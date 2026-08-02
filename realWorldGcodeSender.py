#TODO:  use interpolateCornersCharuco to get better accuracy on corner detection

# import the necessary packages
import numpy as np
import argparse
import cv2
import time
import math
from svgpathtools import Path, Line, QuadraticBezier, CubicBezier, Arc
from svgpathtools import svg2paths, wsvg, svg2paths2, polyline
#import matplotlib
#matplotlib.use('GTK3Agg') 
from matplotlib import pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.backend_bases import MouseButton
from copy import deepcopy
from pygcode import Machine,  GCodeRapidMove, GCodeFeedRate, GCodeLinearMove, GCodeUseMillimeters
import pygcode
from pygcode.gcodes import MODAL_GROUP_MAP
import re

import sys
from svgToGCode import cncPathsClass
from svgToGCode import cncGcodeGeneratorClass
from svgToGCode import Point3D
from svgToGCode import signedArea

from gerbil import Gerbil
import serial.tools.list_ports

import threading
import functools
import queue
import traceback

# Workpiece-frame + probing seam (PROBING_DESIGN.md): pure modules, no hardware deps
from workpiece_frame import Measured, Source, WorkpieceFrame, ZSurface
from probing.base import get_strategy
from probing.targets import propose_z_targets
import probing.strategies  # noqa: F401  (registers z_touch_off / z_mesh / edge_refine)
from toolpath_warp import warp_gcode_lines

# Import configuration system
from app_config import get_config

# Load configuration
config = get_config()

# Extract configuration values for backward compatibility
boxWidth = config.physical_setup.box_width
bedSize = config.get_bed_size()
rightBoxRef = config.get_right_box_ref()
leftBoxRef = config.get_left_box_ref()
rightSlope = config.get_right_slope()
leftSlope = config.get_left_slope()
materialThickness = config.cutting_parameters.material_thickness
cutterDiameter = config.cutting_parameters.cutter_diameter
bedViewSizePixels = config.vision_settings.bed_view_size_pixels


def refresh_config_globals():
    """Refresh legacy module globals after the shared config is edited.

    The original application reads these names throughout its vision and send
    pipeline.  Keeping this small compatibility seam lets the Qt settings view
    apply changes without restarting or duplicating the derived-value math.
    """
    global config, boxWidth, bedSize, rightBoxRef, leftBoxRef
    global rightSlope, leftSlope, materialThickness, cutterDiameter
    global bedViewSizePixels

    config = get_config()
    boxWidth = config.physical_setup.box_width
    bedSize = config.get_bed_size()
    rightBoxRef = config.get_right_box_ref()
    leftBoxRef = config.get_left_box_ref()
    rightSlope = config.get_right_slope()
    leftSlope = config.get_left_slope()
    materialThickness = config.cutting_parameters.material_thickness
    cutterDiameter = config.cutting_parameters.cutter_diameter
    bedViewSizePixels = config.vision_settings.bed_view_size_pixels

#First ID is upper right, which is most positive Z and most positice Y
# Z, Y
global idToLocDict
idToLocDict = {0 :[2,21],
               1 :[2,19],
               2 :[2,17],
               3 :[2,15],
               4 :[2,13],
               5 :[2,11],
               6 :[2, 9],
               7 :[2, 7],
               8 :[2, 5],
               9 :[2, 3],
               10:[2, 1],
               11:[1, 20],
               12:[1, 18],
               13:[1, 16],
               14:[1, 14],
               15:[1, 12],
               16:[1, 10],
               17:[1,  8],
               18:[1,  6],
               19:[1,  4],
               20:[1,  2],
               21:[1,  0],
               22:[0,  21],
               23:[0,  19],
               24:[0,  17],
               25:[0,  15],
               26:[0,  13],
               27:[0,  11],
               28:[0,   9],
               29:[0,   7],
               30:[0,   5],
               31:[0,   3],
               32:[0,   1],
               33:[0,  20],
               34:[0,  18],
               35:[0,  16],
               36:[0,  14],
               37:[0,  12],
               38:[0,  10],
               39:[0,   8],
               40:[0,   6],
               41:[0,   4],
               42:[0,   2],
               43:[0,   0],
               44:[1,  21],
               45:[1,  19],
               46:[1,  17],
               47:[1,  15],
               48:[1,  13],
               49:[1,  11],
               50:[1,   9],
               51:[1,   7],
               52:[1,   5],
               53:[1,   3],
               54:[1,   1],
               55:[2,  20],
               56:[2,  18],
               57:[2,  16],
               58:[2,  14],
               59:[2,  12],
               60:[2,  10],
               61:[2,   8],
               62:[2,   6],
               63:[2,   4],
               64:[2,   2],
               65:[2,   0]}


####################################################################################
# Should put these in a shared libary
####################################################################################
def distanceXY(p1, p2):
  return ((p1.X - p2.X)**2 + (p1.Y - p2.Y)**2)**0.5


def lineOrCurveToPoints3D(lineOrCurve, pointsPerCurve):
  if isinstance(lineOrCurve,Line):
    #print(lineOrCurve)
    return [Point3D(lineOrCurve.bpoints()[0].real, lineOrCurve.bpoints()[0].imag), \
            Point3D(lineOrCurve.bpoints()[1].real, lineOrCurve.bpoints()[1].imag)]
  elif isinstance(lineOrCurve, CubicBezier):
    points3D = []
    for i in range(int(pointsPerCurve)):
      complexPoint = lineOrCurve.point(i / (pointsPerCurve - 1.0))
      points3D.append(Point3D(complexPoint.real, complexPoint.imag, None))
    return points3D
  elif isinstance(lineOrCurve, Arc):
    points3D = []
    for i in range(int(pointsPerCurve) * 10):
      complexPoint = lineOrCurve.point(i / (pointsPerCurve * 10 - 1.0))
      points3D.append(Point3D(complexPoint.real, complexPoint.imag, None))
    return points3D

  else:
    print("unsuported type: " + str(lineOrCurve))
    quit()

def pathToPoints3D(path, pointsPerCurve):
  prevEnd = None
  points3D = []
  for lineOrCurve in path:
    curPoints3D = lineOrCurveToPoints3D(lineOrCurve, pointsPerCurve)
    #check that the last line ending starts the beginning of the next line.
    #print(curPoints3D)
    if prevEnd != None and distanceXY(curPoints3D[0], prevEnd) > 0.01:
      print(curPoints3D[0])
      print(prevEnd)
      print("A SVG path must be contiguous, one line ending and beginning on the next line.  Make a seperate path out of non contiguous lines")
      quit()
    elif prevEnd == None:
      #first line, store both beginning point and end point
      points3D.extend(curPoints3D)
    else:
      #add to point list except first one as it was verified to be same as ending of last
      points3D.extend(curPoints3D[1:])
    prevEnd = curPoints3D[-1]
  return points3D

def centers(x1, y1, x2, y2, r):
    q = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    x3 = (x1 + x2) / 2
    y3 = (y1 + y2) / 2

    xx = (r ** 2 - (q / 2) ** 2) ** 0.5 * (y1 - y2) / q
    yy = (r ** 2 - (q / 2) ** 2) ** 0.5 * (x2 - x1) / q
    return ((x3 + xx, y3 + yy), (x3 - xx, y3 - yy))

def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)

def arcToPoints2(startX, startY, endX, endY, midX, midY):
    points = []
    (centerX, centerY), radius = define_circle((startX, startY), (midX, midY), (endX, endY))
    startAngle = math.atan2(startY - centerY, startX - centerX)
    endAngle   = math.atan2(endY   - centerY, endX   - centerX)
    counterClockWise = 1
    if signedArea([Point3D(startX, startY), Point3D(midX, midY), Point3D(endX, endY)]) < 0:
        counterClockWise = -1
    #for angle in np.arange(startAngle, endAngle, (clockWise * -2 + 1) * 0.1):
    angle = startAngle
    stop = False
    while stop == False:
        x = math.cos(angle) * radius + centerX
        y = math.sin(angle) * radius + centerY
        points.append(Point3D(x, y))
        prevAngle = angle
        angle = (angle +  counterClockWise * 0.1) % (2 * math.pi)
        angleDiff = ((endAngle - angle + 3 * math.pi) % (2 * math.pi)) - math.pi
        #time.sleep(0.05)
        #print(angleDiff)
        if counterClockWise == -1:
          stop = angleDiff <= 0.1 and angleDiff >=0
        else:
          stop = angleDiff >= -0.1 and angleDiff <= 0

    x = math.cos(endAngle) * radius + centerX
    y = math.sin(endAngle) * radius + centerY
    points.append(Point3D(x, y))
    return points

def arcToPoints(startX, startY, endX, endY, i, j, clockWise, curZ):
    points = []
    centerX = startX + i
    centerY = startY + j
    radius = math.dist([centerX, centerY], [startX, startY])
    startAngle = math.atan2(startY - centerY, startX - centerX)
    endAngle   = math.atan2(endY   - centerY, endX   - centerX)
    for angle in np.arange(startAngle, endAngle, (clockWise * -2 + 1) * 0.1):
        x = math.cos(angle) * radius + centerX
        y = math.sin(angle) * radius + centerY
        points.append(Point3D(x, y, curZ))
    x = math.cos(endAngle) * radius + centerX
    y = math.sin(endAngle) * radius + centerY
    points.append(Point3D(x, y, curZ))
    return points

def rotate(origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy

####################################################################################
# OverlayGcode class
####################################################################################
class OverlayGcode:
    def __init__(self, cv2Overhead, gCodeFile = None, svgFile = None, enableSender = True):
        global bedViewSizePixels
        global bedSize
        
        self.bedViewSizePixels = bedViewSizePixels
        self.bedSize = bedSize
        self.xOffset = 0
        self.yOffset = 0
        self.rotation = 0
        self.cv2Overhead = cv2.cvtColor(cv2Overhead, cv2.COLOR_BGR2RGB)
        self.move = False
        self.previewNextDrawnPoint = False
        self.mouseX = 0
        self.mouseY = 0
        self.camRefCenter = [0,0]

        self.startArc = None
        self.endArc = None

        self.refPlateMeasuredLoc = [0.0, 0.0]
        self.camRefCenter = [0.0, 0.0]

        # Workpiece frame (PROBING_DESIGN.md): set by 'z' touch-off, mesh
        # samples added by 'Z'; send paths warp cut Z to follow it when present.
        self.workpieceFrame = None

        fig, ax = plt.subplots()
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.01, right = 0.99)
        plt.axis([self.bedViewSizePixels,0, self.bedViewSizePixels, 0])
        plt.rcParams['keymap.back'].remove('c') # we use c for circle
        plt.rcParams['keymap.save'].remove('s') # we use s for send
        plt.rcParams['keymap.pan'].remove('p') # we use s for send
        if 'r' in plt.rcParams['keymap.home']:
            plt.rcParams['keymap.home'].remove('r') # we use r for resume after feed hold
        if 'h' in plt.rcParams['keymap.home']:
            plt.rcParams['keymap.home'].remove('h') # we use h to home the machine
        if 'g' in plt.rcParams['keymap.grid']:
            plt.rcParams['keymap.grid'].remove('g') # we use g to send a g code file
        #Generate matplotlib plot from opencv image
        self.matPlotImage = plt.imshow(self.cv2Overhead)
        ###############################################
        # Generate controls for plot
        ###############################################
        #xAxes = plt.axes([0.01, 0.8, 0.2, 0.04])
        #self.xBox = TextBox(xAxes, "xOffset (in)", initial="0")
        #label = self.xBox.ax.get_children()[1] # label is a child of the TextBox axis
        #label.set_position([0.5,1]) # [x,y] - change here to set the position
        #label.set_horizontalalignment('center')
        #label.set_verticalalignment('bottom')
        #self.xBox.on_submit(self.onUpdateXOffset)

        #yAxes = plt.axes([0.01, 0.7, 0.2, 0.04])
        #self.yBox = TextBox(yAxes, "yOffset (in)", initial="0")
        #label = self.yBox.ax.get_children()[1] # label is a child of the TextBox axis
        #label.set_position([0.5,1]) # [x,y] - change here to set the position
        #label.set_horizontalalignment('center')
        #label.set_verticalalignment('bottom')
        #self.yBox.on_submit(self.onUpdateYOffset)

        #rAxes = plt.axes([0.01, 0.6, 0.2, 0.04])
        #self.rBox =  TextBox(rAxes, "rotation (deg)", initial="0")
        #label = self.rBox.ax.get_children()[1] # label is a child of the TextBox axis
        #label.set_position([0.5,1]) # [x,y] - change here to set the position
        #label.set_horizontalalignment('center')
        #label.set_verticalalignment('bottom')
        #self.rBox.on_submit(self.onUpdateRotation)

        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        cid = fig.canvas.mpl_connect('button_release_event', self.onrelease)
        cid = fig.canvas.mpl_connect('motion_notify_event', self.onmousemove)
        cid = fig.canvas.mpl_connect('key_press_event', self.onkeypress)

        #Create object to handle controlling the CNC machine and sending the g codes to it
        self.gCodeFile = gCodeFile
        if enableSender:
            self.sender = GCodeSender()
        

        self.points = []
        self.drawnPoints = []
        self.laserPowers = []
        self.machine = Machine()


        ############################################################################
        # If generating cut paths from an SVG File
        ############################################################################
        self.svgFile = svgFile
        if svgFile != None:
            #Generate cncPaths object based on svgFile
            #These are in mm
            self.cncPaths = cncPathsClass(inputSvgFile   = svgFile,
                                          pointsPerCurve = 30,
                                          distPerTab      = 7.87,
                                          tabWidth        = 0.25,
                                          cutterDiameter  = cutterDiameter,
                                          convertSvfToIn  = True
                                         )

            #Order cuts from inside holes to outside borders
            self.cncPaths.orderCncHolePathsFirst()
            self.cncPaths.orderPartialCutsFirst()

            self.pathIndex = -1
            self.pathOffsets = []
            for path in self.cncPaths.cncPaths:
                self.pathOffsets.append([0.0, 0.0])
                

        elif gCodeFile != None:
            with open(gCodeFile, 'r') as fh:
              for line_text in fh.readlines():
                line = pygcode.Line(line_text)
                prevPos = self.machine.pos
                self.machine.process_block(line.block)
                
                ######################################
                # First determine machine motion mode and power
                ######################################
                motion = str(self.machine.mode.modal_groups[MODAL_GROUP_MAP['motion']])
                sCode = str(self.machine.mode.modal_groups[MODAL_GROUP_MAP['spindle_speed']])
                power = sCode.split('S')[1]
                #Make rapid movements 0 laser power
                if motion == "G00" or motion == "G0":
                  self.laserPowers.append(0.0)
                else:
                  self.laserPowers.append(float(power) / 100.0)

                ######################################
                # Determine machine current unit, convert to inches
                ######################################
                unit = str(self.machine.mode.modal_groups[MODAL_GROUP_MAP['units']])
                x = None
                if unit == "G20":
                  if motion == "G02" or motion == "G2" or motion == "G03" or motion == "G3":
                    beforeComment = line_text.split("(")[0]
                    resultX = re.search('X[+-]?([0-9]*[.])?[0-9]+', beforeComment)
                    resultY = re.search('Y[+-]?([0-9]*[.])?[0-9]+', beforeComment)
                    resultI = re.search('I[+-]?([0-9]*[.])?[0-9]+', beforeComment)
                    resultJ = re.search('J[+-]?([0-9]*[.])?[0-9]+', beforeComment)
                    if resultX != None:
                        x = float(resultX.group()[1:])
                        y = float(resultY.group()[1:])
                        i = float(resultI.group()[1:])
                        j = float(resultJ.group()[1:])
                        clockWise = "G02" in motion or "G2" in motion
                  else:
                    self.points.append(Point3D(self.machine.pos.X, self.machine.pos.Y, self.machine.pos.Z))
                else:
                  if motion == "G02" or motion == "G2" or motion == "G03" or motion == "G3":
                    resultX = re.search('X[+-]?([0-9]*[.])?[0-9]+', beforeComment)
                    resultY = re.search('Y[+-]?([0-9]*[.])?[0-9]+', beforeComment)
                    resultI = re.search('I[+-]?([0-9]*[.])?[0-9]+', beforeComment)
                    resultJ = re.search('J[+-]?([0-9]*[.])?[0-9]+', beforeComment)
                    if resultX != None:
                        x = float(resultX.group()[1:]) / 25.4
                        y = float(resultY.group()[1:]) / 25.4
                        i = float(resultI.group()[1:]) / 25.4
                        j = float(resultJ.group()[1:]) / 25.4
                        clockWise = "G02" in motion or "G2" in motion
                  else:
                    self.points.append(Point3D(self.machine.pos.X / 25.4, self.machine.pos.Y / 25.4, self.machine.pos.Z / 25.4))
                if x != None:
                  self.points.extend(arcToPoints(prevPos.X, prevPos.Y, self.machine.pos.X, self.machine.pos.Y, i, j, clockWise, self.machine.pos.Z))
                  self.laserPowers.extend([self.laserPowers[-1]] * (len(self.points) - len(self.laserPowers)))
        else:
            print("Must use either svg or gCode file")
            quit()

            
        self.updateOverlay()

    def set_ref_loc(self, refPixels):
        print("refPixels: " + str(refPixels))

        refPoints = []
        for refPixel in refPixels:
            refPoints.append( self._pixel_to_inches(refPixel[0], refPixel[1]))
        self.refPoints = refPoints

        avgX = (refPoints[0][0] + refPoints[1][0] + refPoints[2][0] + refPoints[3][0]) / 4.0
        avgY = (refPoints[0][1] + refPoints[1][1] + refPoints[2][1] + refPoints[3][1]) / 4.0
        self.camRefCenter = [avgX, avgY]
        # set measured ref plate location whenever reference plat is moved
        self.refPlateMeasuredLoc = self.camRefCenter + [0]
        print("Ref Points = " + str(refPoints))
        angle = getBoxAngle(refPoints)
        print("Angle: " + str(angle * 180 / math.pi))
        
    
    def phyPointsToPixels(self, transformedPoints):
        global rightBoxRef, leftBoxRef

        # Bed drawn from Y = 0 to Y = 35, but from X at left support beam and right support beam with ref boxes on them.
        self.offsetPoints(transformedPoints, -rightBoxRef.X, 0)
        self.scalePoints(transformedPoints, \
                         self.bedViewSizePixels / (leftBoxRef.X - rightBoxRef.X), \
                         self.bedViewSizePixels / self.bedSize.Y)

    def scalePoints(self, points, scaleX, scaleY):
      for point in points:
        point.X = point.X * scaleX
        point.Y = point.Y * scaleY

    def offsetPoints(self, points, X, Y):
      for point in points:
        point.X = point.X + X
        point.Y = point.Y + Y

    def rotatePoints(self, points, origin, angle):
      for point in points:
        point.X, point.Y = rotate(origin, [point.X, point.Y], angle)

    def overlaySvgOrGcode(self, image, xOff = 0, yOff = 0, rotation = 0):
      """
      image is opencv image
      xOff is in inches
      yOff is in inches
      rotation is in degrees
      """
      global cutterDiameter
      toolWidth = round(abs(cutterDiameter * self.bedViewSizePixels / (leftBoxRef.X - rightBoxRef.X)))
      #convert to radians
      rotation = rotation * math.pi / 180
      overlay = image.copy()

      if self.gCodeFile != None:
          #Make copy of points before transforming them
          transformedPoints = deepcopy(self.points)
          self.offsetPoints(transformedPoints, xOff, yOff)
          self.rotatePoints(transformedPoints, [xOff, yOff], rotation)
      else:
          transformedPoints = []
          self.laserPowers = []
          for cncPath, offset in zip(self.cncPaths.cncPaths, self.pathOffsets):
              newPoints = deepcopy(cncPath.points3D)

              self.offsetPoints(newPoints, offset[0] , offset[1])
              transformedPoints.extend(newPoints)
              if cncPath.color[1] == 0:
                  self.laserPowers.extend([0] + [1] * (len(cncPath.points3D) - 1))
              else:
                  self.laserPowers.extend([cncPath.color[1] / 255] * len(cncPath.points3D))
          self.rotatePoints(transformedPoints, [offset[0], offset[1]], rotation)

      #Then convert to pixel location
      self.phyPointsToPixels(transformedPoints)
      prevPoint = None
      for point, laserPower in zip(transformedPoints, self.laserPowers):
        newPoint = (int(point.X), int(point.Y))
        if prevPoint is not None:
          cv2.line(overlay, prevPoint, newPoint, (int(laserPower * 255), 0, 0), toolWidth)
        prevPoint = newPoint

      transformedPoints = deepcopy(self.drawnPoints)
      self.phyPointsToPixels(transformedPoints)
      prevPoint = None
      for point in transformedPoints:
        newPoint = (int(point.X), int(point.Y))
        if prevPoint is not None:
          cv2.line(overlay, prevPoint, newPoint, (0, 255, 0), toolWidth)
        prevPoint = newPoint
      return overlay

    def updateOverlay(self):
        overlay = self.overlaySvgOrGcode(self.cv2Overhead, self.xOffset, self.yOffset, self.rotation)
        self.matPlotImage.set_data(overlay)
        self.matPlotImage.figure.canvas.draw()

    def onUpdateXOffset(self, text):
      if self.xOffset == float(text):
        return
      self.xOffset = float(text)
      self.updateOverlay()
      
    def onUpdateYOffset(self, text):
      if self.yOffset == float(text):
        return
      self.yOffset = float(text)
      self.updateOverlay()

    def onUpdateRotation(self, text):
      if self.rotation == float(text):
        return
      self.rotation = float(text)
      self.updateOverlay()

    def onmousemove(self, event):
      self.move = True
      self.mouseX = event.xdata
      self.mouseY = event.ydata
      if self.previewNextDrawnPoint:
        # if finishing drawing an arch, then preview that instead of a line
        if self.startArc != None and self.endArc != None:
            x, y = self._mouse_pos_inches()
            radius = math.dist([x,y], [self.startArc.X, self.startArc.Y])
            newPoints = arcToPoints2(self.startArc.X, self.startArc.Y, self.endArc.X, self.endArc.Y, x, y)
            print(newPoints)
            #newPoints = arcToPoints(self.startArc.X, self.startArc.Y, self.endArc.X, self.endArc.Y, x - self.startArc.X, y - self.startArc.Y, False, 0)
            numNewPoints = len(newPoints)
            self.drawnPoints.extend(newPoints)
            self.updateOverlay()
            #remove added points as this is just a preview
            del self.drawnPoints[-numNewPoints:]
            

        else:
            x, y = self._mouse_pos_inches()
            #provision to preview first point
            firstPoint = False
            if len(self.drawnPoints) == 0:
              self.drawnPoints.append(Point3D(x, y, 0))
              firstPoint = True
            self.drawnPoints.append(Point3D(x, y, 0))
            self.updateOverlay()
            self.drawnPoints.pop()
            if firstPoint:
              self.drawnPoints.pop()

    def _pixel_to_inches(self, x, y):
      x = x / self.bedViewSizePixels * (leftBoxRef.X - rightBoxRef.X)
      x = x + rightBoxRef.X
      y = y / self.bedViewSizePixels * self.bedSize.Y
      # Make reference plate measured center location in image be exact measured location
      x = x - self.camRefCenter[0] + self.refPlateMeasuredLoc[0]
      y = y - self.camRefCenter[1] + self.refPlateMeasuredLoc[1]
      return x, y
    def _mouse_pos_inches(self):
      print(self.mouseX)
      print(self.bedViewSizePixels)
      return self._pixel_to_inches(self.mouseX, self.mouseY)

    def _workCoordPaths(self):
        """Current cut paths as (x, y) polylines in the work coordinates the
        send paths will emit them in. Used to propose probe targets."""
        paths = []
        if self.drawnPoints:
            # 'C' sends drawn points offset by the workspace zero
            rx, ry = self.refPlateMeasuredLoc[0], self.refPlateMeasuredLoc[1]
            paths.append([(p.X - rx, p.Y - ry) for p in self.drawnPoints])
        if self.svgFile is not None:
            # mirror the transforms 's' applies before send_svf
            rotation = self.rotation * math.pi / 180
            origin = [self.pathOffsets[-1][0], self.pathOffsets[-1][1]]
            for path, offset in zip(self.cncPaths.cncPaths, self.pathOffsets):
                pts = deepcopy(path.points3D)
                self.offsetPoints(pts, offset[0], offset[1])
                self.rotatePoints(pts, origin, rotation)
                paths.append([(p.X, p.Y) for p in pts])
        elif self.gCodeFile is not None and self.points:
            # file coords; the controller applies the G54/G68 placement
            paths.append([(p.X, p.Y) for p in self.points])
        return paths

    def onkeypress(self, event):
        try:
            x, y = self._mouse_pos_inches()
        except (TypeError, AttributeError):
            # mouse is off the canvas (or never entered it); keys that need a
            # position check for None below -- recovery keys must still work
            x, y = None, None
        print(event.key)
        # if sending a g code file
        if   event.key == 'g':
            self.sender.run_async('send_file', self.sender.send_file,
                                  self.gCodeFile, self.xOffset, self.yOffset, self.rotation)
        # if sending an svf file
        elif event.key == 's':
            # Bail early if a send is already running: the transforms below
            # mutate the path points in place and must not be applied twice.
            if self.sender.is_busy():
                print("CNC busy; ignoring 's'")
                return
            # perform transfomrations that were done in GUI on actual points in the path
            rotation = self.rotation * math.pi / 180
            for path, offset in zip(self.cncPaths.cncPaths, self.pathOffsets):
              print("before: " + str(path.points3D[0]) + " offset: " + str(offset))
              self.offsetPoints(path.points3D, offset[0] , offset[1])
              self.rotatePoints(path.points3D, [self.pathOffsets[-1][0], self.pathOffsets[-1][1]], rotation)
              print("after: " + str(path.points3D[0]))

            # Before sending add tabs
            #self.cncPaths.addTabs()
            #self.cncPaths.ModifyPointsFromTabLocations()

            self.sender.run_async('send_svf', self.sender.send_svf,
                                  self.cncPaths, self.workpieceFrame)
        # select next/previous svg path (-1 selects all paths)
        elif event.key == 'n':
            if self.svgFile is not None:
                self.pathIndex = min(self.pathIndex + 1,
                                     len(self.cncPaths.cncPaths) - 1)
                print("Selected path: " + str(self.pathIndex))
        elif event.key == 'p':
            if self.svgFile is not None:
                self.pathIndex = max(self.pathIndex - 1, -1)
                print("Selected path: " + str(self.pathIndex))

        elif event.key == 'h':
            # home_machine() sends immediately, so guard against injecting a
            # homing cycle into a running probe/job.
            if self.sender.is_busy():
                print("CNC busy; ignoring 'h'")
                return
            self.sender.home_machine()

        elif event.key == 'z':
            #Find X, Y, and Z position of the aluminum reference block on the work piece
            #sepcify the X and Y estimated position of the reference block
            #self.refPlateMeasuredLoc = self.sender.zero_on_refPlate(self.refPoints)
            #Just probe z height for now for demo
            def zero_and_frame():
                self.sender.zero_on_refPlate(self.refPoints, True)
                # Work zero now sits on the probed surface: record an identity
                # frame (workpiece coords == work coords) with Z at PROBE
                # fidelity, the spine 'Z' mesh probing and send-time warping
                # ride on (PROBING_DESIGN.md).
                self.workpieceFrame = WorkpieceFrame(
                    x=Measured(0.0, Source.PROBE, tolerance=0.002),
                    y=Measured(0.0, Source.PROBE, tolerance=0.002),
                    angle=Measured(0.0, Source.PROBE),
                    z=ZSurface(nominal=Measured(0.0, Source.PROBE, tolerance=0.001)))
            self.sender.run_async('zero_on_refPlate', zero_and_frame)
            print("refPlateMeasuredLoc: " + str(self.refPlateMeasuredLoc))
            print("camRefCenter: " + str(self.camRefCenter))

        elif event.key == 'Z':
            # Probe a Z mesh over the current cut region so send-time warping
            # follows the real surface (constant depth of cut on warped stock)
            if self.sender.is_busy():
                print("CNC busy; ignoring 'Z'")
                return
            paths = self._workCoordPaths()
            if not paths:
                print("No cut paths loaded or drawn; nothing to mesh-probe")
                return
            targets = propose_z_targets(paths)
            if not targets:
                print("Could not propose any probe targets")
                return
            if self.workpieceFrame is None:
                # Identity frame: workpiece coords == work coords. Meaningful
                # once 'z' touch-off has set work zero on the stock top.
                self.workpieceFrame = WorkpieceFrame.eyeballed(0.0, 0.0)
            print("Probing {}-point Z mesh".format(len(targets)))
            def run_mesh():
                self.sender.flushGcodeRespQue()
                machine = GCodeSenderMachine(self.sender)
                self.workpieceFrame = get_strategy("z_mesh").refine(
                    self.workpieceFrame, machine, targets)
                print("Z mesh samples: " + str(self.workpieceFrame.z.samples))
            self.sender.run_async('z_mesh', run_mesh)

        elif event.key == 'm':
            # absolute_move() sends immediately, so guard against interleaving
            # a jog with a running probe/job.
            if self.sender.is_busy():
                print("CNC busy; ignoring 'm'")
                return
            if x is None:
                print("Mouse not over the bed view; ignoring 'm'")
                return
            self.sender.absolute_move(x, y, feed = 300)

        # -- recovery / safety keys: deliberately NOT guarded by is_busy(),
        # they exist precisely for when an operation is running or stuck.
        elif event.key == ' ':
            print("FEED HOLD (!)")
            self.sender.gerbil.hold()
        elif event.key == 'r':
            print("Resume (~)")
            self.sender.gerbil.resume()
        elif event.key == 'x':
            print("Kill alarm ($X)")
            self.sender.gerbil.killalarm()

        elif event.key == 'd':
            if x is None:
                return
            # first d turns on preview
            if not self.previewNextDrawnPoint:
                self.previewNextDrawnPoint = True
                return
            self.drawnPoints.append(Point3D(x, y, 0))
            self.updateOverlay()
            print(self.drawnPoints)
        elif event.key == 'c':
          if x is None:
              return
          if self.startArc == None:
              self.startArc = self.drawnPoints[-1]#Point3D(x, y)
              self.endArc = Point3D(x, y)
          else:
              print(self.startArc)
              print(self.endArc)
              newPoints = arcToPoints2(self.startArc.X, self.startArc.Y, self.endArc.X, self.endArc.Y, x, y)
              print("newPoints:" + str(newPoints))
              self.drawnPoints.extend(newPoints)
              self.startArc = None
              self.endArc = None
              self.updateOverlay()
        # erase one drawn point
        elif event.key == 'e':
            if len(self.drawnPoints) > 0:
                self.drawnPoints.pop()
            self.updateOverlay()
        # Erase all drawn points
        elif event.key == 'E':
            self.drawnPoints = []
            self.updateOverlay()
        elif event.key == 'C':
            # offset G codes by workspace zero as G codes send relative to workspace offset
            offset = Point3D(-self.refPlateMeasuredLoc[0], \
                             -self.refPlateMeasuredLoc[1])
            self.sender.run_async('send_drawnPoints',
                                  self.sender.send_drawnPoints, offset,
                                  self.drawnPoints, self.workpieceFrame)
        elif event.key == 'shift':
            self.shiftHeld = True
            print("shift")

        # if a non drawing key was pushed then exit drawing preveiw mode
        if event.key != 'd' and event.key.lower() != 'e' and event.key != 'shift' and event.key != 'c':
            if self.previewNextDrawnPoint:
                self.previewNextDrawnPoint = False
                self.updateOverlay()
        if event.key != 'c':
            self.startArc = None
            self.endArc = None


    def onclick(self, event):
      self.move = False

    def onrelease(self, event):
      global matPlotImage
      #If clicking outside region, or mouse moved since released then return
      
      if event.x < 260 or self.move == True:
        return
      # click landed outside the bed axes (figure margin): no position to use
      if event.xdata is None or self.mouseX is None:
        return
      pixelsToOrigin = np.array([event.xdata, event.ydata])
      print("event x,y: " + str(pixelsToOrigin))
      print("mouse x,y: " + str([self.mouseX, self.mouseY]))
      xIn, yIn = self._mouse_pos_inches()
      if event.button == MouseButton.RIGHT:
          self.rotation = math.atan2(yIn - self.yOffset, xIn - self.xOffset)
          self.rotation = self.rotation - math.pi/2.0
          self.rotation = self.rotation * 180 / math.pi

      else:
          self.xOffset = xIn
          self.yOffset = yIn
          print("xin, yIn: " + str(xIn) + "," + str(yIn))
          # per-path offsets only exist in svg mode; gcode mode places the
          # whole file with xOffset/yOffset alone
          if self.svgFile is not None:
              # if negative 1 then apply offset to all paths, else just selected path
              if self.pathIndex == -1:
                  for i in range(len(self.pathOffsets)):
                      self.pathOffsets[i] = [xIn, yIn]
              else:
                  minX = 1000000000
                  minY = 1000000000
                  for point in self.cncPaths.cncPaths[self.pathIndex].points3D:
                      minX = min(minX, point.X)
                      minY = min(minY, point.Y)
                  self.pathOffsets[self.pathIndex]  = [xIn - minX, yIn - minY]
      self.updateOverlay()

def crop_half_vertically(img):
  #cropped_img = image[,int(image.shape[1]/2):int(image.shape[1])]
  #height = img.shape[0]
  width = img.shape[1]
  # Cut the image in half
  width_cutoff = int(width // 2)
  left = img[:, :width_cutoff]
  right = img[:, width_cutoff:]
  return left, right

def getBoxAngle(points):
    adjacentPoints = []
    maxDistance = 0
    for point in points[1:]:
        maxDistance = max(maxDistance, math.dist(point, points[0]))
    for point in points[1:]:
        # if an adjacent point (not across from box)
        if math.dist(point, points[0]) != maxDistance:
            print("adjacen points:")
            print(points[0])
            print(point)
            angle = math.atan2(point[1] - points[0][1], point[0] - points[0][0])
            break
    print(angle * 180 / math.pi)
    # a square is square, so we will pick one of the 4 angles 0, + 90, +180, or +270
    if angle < math.pi / 4:
        angle = angle + math.pi
    if angle >= math.pi / 4:
        angle = angle - math.pi / 2
    return angle

def sortBoxPoints(points, rightSide = True):
  #First sort by X
  sortedX = sorted(points , key=lambda k: [k[0]])
  #Then sorty by Y left and right most two X set of points
  rightTwoPoints = sorted(sortedX[2:], key=lambda k: [k[1]])
  leftTwoPoints  = sorted(sortedX[0:2], key=lambda k: [k[1]])
  if rightSide:
    minZminY = leftTwoPoints[1]
    minZmaxY = leftTwoPoints[0]
    maxZmaxY = rightTwoPoints[0]
    maxZminY = rightTwoPoints[1]
  else:
    minZminY = rightTwoPoints[1]
    minZmaxY = rightTwoPoints[0]
    maxZmaxY = leftTwoPoints[0]
    maxZminY = leftTwoPoints[1]
    
  return [minZminY, minZmaxY, maxZmaxY, maxZminY]
def get_id_loc(image, boxes, ids, ID):
    for box, curID in zip(boxes, ids):
        if curID != ID:
            continue
        boxPoints = box[0]
        boxPointsSorted = np.array(sortBoxPoints(boxPoints))
        return boxPointsSorted
    return None

def boxes_to_point_and_location_list(boxes, ids, image, rightSide = False):
  global boxWidth
  global idToLocDict
  pointList = []
  locations = []
  for box, ID in zip(boxes, ids):
    #IDs below 33 are on right side, skip those if looking for left side points
    if (rightSide == False and ID < 33) or \
       (rightSide == True  and ID >= 33) or \
       (ID > 65):
      continue
    boxLoc = idToLocDict[ID[0]]
    for boxPoints in box:
      prevX = int(boxPoints[0][0])
      prevY = int(boxPoints[0][1])
      i = 0
   
      font                   = cv2.FONT_HERSHEY_SIMPLEX
      bottomLeftCornerOfText = prevX + 100,prevY
      fontScale              = 1
      fontColor              = (0,255,255)
      thickness              = 3
      lineType               = 2

      cv2.putText(image,str(ID[0]), 
          bottomLeftCornerOfText, 
          font, 
          fontScale,
          fontColor,
          thickness,
          lineType)

      boxPointsSorted = sortBoxPoints(boxPoints, rightSide)
      for point in boxPointsSorted:
        ############################################
        # Generate list of points
        ############################################
        pointList.append(point)

        
        ############################################
        # Generate point location based on boxWidth and index within box
        ############################################
        curLoc = [0,0]
        if i == 0:
          curLoc[0] = boxLoc[0] + 0
          curLoc[1] = boxLoc[1] + 0
        elif i ==1:
          curLoc[0] = boxLoc[0] + 0
          curLoc[1] = boxLoc[1] + 1
        elif i == 2:
          curLoc[0] = boxLoc[0] + 1
          curLoc[1] = boxLoc[1] + 1
        else:
          curLoc[0] = boxLoc[0] + 1
          curLoc[1] = boxLoc[1] + 0
        if rightSide:
          curLoc[0] = curLoc[0] * boxWidth + rightBoxRef.Z + curLoc[1] * rightSlope
          curLoc[1] = curLoc[1] * boxWidth + rightBoxRef.Y
        else:
          curLoc[0] = curLoc[0] * boxWidth + leftBoxRef.Z + curLoc[1] * leftSlope
          curLoc[1] = curLoc[1] * boxWidth + leftBoxRef.Y
        locations.append(curLoc)

        ############################################
        # Display points on image
        ############################################
        x= int(point[0])
        y= int(point[1])
        image = cv2.arrowedLine(image, (prevX,prevY), (x,y),
                                (0,255,255), 3)

        prevX = x
        prevY = y
        i = i + 1
        
  return np.array(pointList), locations, image

def generate_dest_locations(corners, image):
  global boxWidth
  prevX=2000
  prevY=2000
  locations = []
  yIndex = 0
  xIndex = 0
  for corner in corners:
    x,y= corner
    x= int(x)
    y= int(y)

    #cv2.rectangle(gray, (prevX,prevY),(x,y),(i*3,0,0),-1)
    image = cv2.arrowedLine(image, (prevX,prevY), (x,y),
                                     (200,0,0), 5)
    locations.append([xIndex * boxWidth, yIndex * boxWidth])
    if xIndex == 2:
      xIndex = 0
      yIndex = yIndex + 1
    else:
      xIndex = xIndex + 1
    prevX = x
    prevY = y
  return locations, image

def display_4_lines(pixels, frame, flip=False):
  line1 = tuple(pixels[0][0].astype(int))
  line2   = tuple(pixels[1][0].astype(int))
  if flip:
    line3   = tuple(pixels[3][0].astype(int))
    line4   = tuple(pixels[2][0].astype(int))
  else:
    line3   = tuple(pixels[2][0].astype(int))
    line4   = tuple(pixels[3][0].astype(int))
  cv2.line(frame, line1,line2,(0,255,255),3)
  cv2.line(frame, line2,line3,(0,255,255),3)
  cv2.line(frame, line3,line4,(0,255,255),3)
  cv2.line(frame, line4,line1,(0,255,255),3)

class GCodeSender:
    def __init__(self):
        # Thread-safe queue of responses read back from the controller.  The
        # gerbil callback runs on the serial reader thread and pushes onto this
        # queue; waitOnGCodeComplete() blocks on it from whichever thread is
        # running a CNC operation.
        self.respQueue = queue.Queue()

        # Background worker used to run long blocking operations (probing,
        # streaming a file, etc.) off the matplotlib GUI thread so the UI stays
        # responsive.  Only one operation is allowed to run at a time.
        self._job_thread = None
        self._job_lock = threading.Lock()

        # Read-only observers (the Qt bridge) receive copies of every Gerbil
        # event.  They must never consume respQueue, which remains owned by the
        # blocking machine protocols below.
        self._event_listeners = []
        self._last_boot_banner = None

        self.gerbil = Gerbil(self.gerbil_callback)
        self.gerbil.setup_logging()

        ports = serial.tools.list_ports.comports()
        for p in ports:
            print(p.device)

        self.gerbil.cnect(config.communication_settings.com_port, config.communication_settings.baud_rate)
        self.gerbil.poll_start()
        self.set_inches()

        self.plateHeight = config.probing_settings.plate_height
        self.plateWidth  = config.probing_settings.plate_width
        self.cutterRadius   = config.cutting_parameters.cutter_diameter / 2.0
        self.distToKnotch = config.probing_settings.dist_to_notch




    def add_event_listener(self, listener):
        """Subscribe to Gerbil events without changing existing queue logic."""
        if listener not in self._event_listeners:
            self._event_listeners.append(listener)
        try:
            if self._last_boot_banner is not None:
                listener("on_read", self._last_boot_banner)
            if self.gerbil.cmode is not None:
                listener("on_stateupdate", self.gerbil.cmode,
                         tuple(self.gerbil.cmpos), tuple(self.gerbil.cwpos))
        except Exception:
            traceback.print_exc()
        return listener

    def remove_event_listener(self, listener):
        try:
            self._event_listeners.remove(listener)
        except ValueError:
            pass

    def gerbil_callback(self, eventstring, *data):
        if eventstring == "on_read" and data and "Grbl " in str(data[0]):
            self._last_boot_banner = str(data[0])
        for listener in tuple(self._event_listeners):
            try:
                try:
                    listener_data = deepcopy(data)
                except Exception:
                    listener_data = tuple(data)
                listener(eventstring, *listener_data)
            except Exception:
                # A UI listener runs on the serial reader thread.  Never let a
                # view bug terminate that thread or starve machine protocols.
                traceback.print_exc()

        args = []
        #if eventstring != 'on_vars_change' and \
        #   eventstring != 'on_progress_percent' and \
        #   eventstring != 'on_log' and \
        #   eventstring != 'on_write' and \
        #   eventstring != 'on_line_sent' and \
        #   eventstring != 'on_bufsize_change':
        #   print("GERBIL CALLBACK: " + eventstring)
        if eventstring != "on_read":
            return
        print()
        print()
        for d in data:
            args.append(str(d))
            print(d)
        print("args    event={} data={}".format(eventstring.ljust(30), ", ".join(args)))
        self.curData = data
        self.curEvent = eventstring

        # Hand the response off to any waiter.  Queue.put() is thread-safe and
        # wakes a blocked waitOnGCodeComplete() without busy-spinning.
        self.respQueue.put(data)

    def get_absolute_pos(self):
        self.gerbil.send_immediately("?\n")
        # Status reports come back near-instantly; a long silence here means
        # comms are dead, so fail fast rather than waiting the full default.
        resp = self.waitOnGCodeComplete(">", timeout = 10.0)
        m = re.match("<(.*?),MPos:(.*?),WPos:(.*?)>", resp)
        mpos_parts = m.group(2).split(",")
        return (float(mpos_parts[0]), float(mpos_parts[1]), float(mpos_parts[2]))

    def home_machine(self):
        self.gerbil.send_immediately("$H\n")
        pass

    def set_work_coord_offset(self, x = None, y = None, z = None):
        xStr, yStr, zStr = self._get_xyz_string(x, y, z)
        self.gerbil.send_immediately("G54 " + xStr + yStr + zStr + "\n")
        self.gerbil.send_immediately("G54\n")

    def set_cur_pos_as(self, x = None, y = None, z = None):
        xStr, yStr, zStr = self._get_xyz_string(x, y, z)
        self.gerbil.send_immediately("G92 " + xStr + yStr + zStr + "\n")

    def probe(self, x = None, y = None, z = None, feed = 5.9):
        xStr, yStr, zStr = self._get_xyz_string(x, y, z)
        self.gerbil.send_immediately("G38.2 " + xStr + yStr + zStr + " F" + str(feed) + "\n")
        PrbResp = self.waitOnGCodeComplete("PRB")
        # Example output:  '[PRB:-0.1965,-0.1965,-2.0697:1]'
        print("PrbResp: " + str(PrbResp))
        tmp = PrbResp.split("PRB:")[1]
        tmp = tmp.split(":1")[0]
        numbers = tmp.split(",")
        print("Numbers: " + str(numbers))
        return [float(x) for x in numbers]

    def probeZSequence(self):
        plateHeight = self.plateHeight
        #Move down medium speed to reference plate
        print("***************************************4")
        print("***************************************5")
        self.probe(z = -2.75, feed = config.probing_settings.probe_feed_rate_fast) # move down by 2.75" until probe hit
        print("***************************************6")
        self.set_cur_pos_as(z = plateHeight) # Set actual 0 to probed location
        print("***************************************7")

        #Move up, then slowly to reference plate
        self.work_offset_move(z = plateHeight + 0.1, feed=180) # Move just above reference plate
        xyz = self.probe(z = plateHeight-0.05, feed = config.probing_settings.probe_feed_rate_slow)
        self.set_cur_pos_as(z = plateHeight) # Set actual 0 to probed location
        self.work_offset_move(z = plateHeight + 0.5, feed=180) # Move just above reference plate, clearing lip on reference plate
        # return z height of the probe
        return xyz[0], xyz[1], xyz[2] - plateHeight


    def probeXYSequence(self, plateAngle):
        cutterRadius = self.cutterRadius
        # Firxt xAxis then yAxis
        for axisAngle in [0, math.pi / 2.0]:
            angle = axisAngle + plateAngle
            
            plateHeight = self.plateHeight
            plateWidth = self.plateWidth # total width of touch plate...half of this is distance to center of touch plate
            firstSafeDist = plateWidth * 0.75
            secSafeDist = plateWidth * 0.5 + cutterRadius + 0.1 # can get a little closer second time as we already sensed edge of plate once
            probeToDist = plateWidth * 0.25

            # Move up
            self.work_offset_move(z = plateHeight + 0.5, feed=100) # Move just above reference plate, clearing lip on reference plate
            #once medium speed, once slow speed
            for feed, dist in zip([config.probing_settings.probe_feed_rate_fast, config.probing_settings.probe_feed_rate_slow], [firstSafeDist, secSafeDist]):
                # Move to side of touch plate
                self.work_offset_move(x = math.cos(angle) * dist, y = math.sin(angle) * dist, feed=180)
                # Move below touch plate
                self.work_offset_move(z = plateHeight-0.1, feed=100)
                # Probe to the touch plate
                refPoint = self.probe(x = math.cos(angle) * probeToDist, y = math.sin(angle) * probeToDist, feed=config.probing_settings.probe_feed_rate_fast)
                # set this as new side of touch plate
                self.set_cur_pos_as(x = math.cos(angle) * (plateWidth * 0.5 + cutterRadius) , \
                                           y = math.sin(angle) * (plateWidth * 0.5 + cutterRadius))

            # Move away from plate and up, then to center of touchplate
            self.work_offset_move(x = math.cos(angle) * secSafeDist, y = math.sin(angle) * secSafeDist, feed = 100)
            self.work_offset_move(z = plateHeight + 0.5, feed=180) # Move just above reference plate, clearing lip on reference plate
        
        # we want work coord system to be center of knotch of touch plate, not center of touch plate itself.  Move there then make that zero.
        self.work_offset_move(x = math.cos(angle - math.pi/4.0) * self.distToKnotch, y = math.sin(angle - math.pi/4.0) * self.distToKnotch, feed = 400)
        self.set_cur_pos_as(x = 0, y = 0)
        return [refPoint[0] - math.cos(angle) * (plateWidth * 0.5 + cutterRadius) + math.cos(angle - math.pi/4.0) * self.distToKnotch, \
                refPoint[1] - math.sin(angle) * (plateWidth * 0.5 + cutterRadius) + math.sin(angle - math.pi/4.0) * self.distToKnotch]


    def probeAngleOfTouchPlate(self, estPlateAngle, x, y):
        plateHeight = self.plateHeight
        plateWidth = self.plateWidth # total width of touch plate...half of this is distance to center of touch plate
        firstSafeDist = plateWidth * 0.75
        secSafeDist = plateWidth * 0.625 # can get a little closer second time as we already sensed edge of plate once
        probeToDist = plateWidth * 0.25


        angle = estPlateAngle 
        # Move up
        self.work_offset_move(z = plateHeight + 0.5, feed=180) # Move just above reference plate, clearing lip on reference plate
        # Move to side of touch plate and down a quarter of the plate width

        # this routine does not adjust work offset, so need to always be conservative
        #Probe down a quarter of touch plate first
        #once medium speed, once slow speed
        distAdjust = 0
        for feed, dist in zip([config.probing_settings.probe_feed_rate_fast, config.probing_settings.probe_feed_rate_slow], [firstSafeDist, firstSafeDist]):
            self.work_offset_move(x = math.cos(angle) * (dist - distAdjust) + math.cos(angle - math.pi/2.0) * plateWidth * 0.25 , \
                                  y = math.sin(angle) * (dist - distAdjust) + math.sin(angle - math.pi/2.0) * plateWidth * 0.25, feed=400)
            # Move below touch plate
            self.work_offset_move(z = plateHeight-0.1, feed=180)
            # Probe to the touch plate
            ref1 = self.probe(x = math.cos(angle) * probeToDist + math.cos(angle - math.pi/2.0) * plateWidth * 0.25 , \
                              y = math.sin(angle) * probeToDist + math.sin(angle - math.pi/2.0) * plateWidth * 0.25, feed=feed)
            # move just 0.1" away from plate next probe since we know where idge roughy is now
            distAdjust = dist - math.dist([x,y], [ref1[0], ref1[1]]) - 0.05
            print(ref1)
            print("distAdjust: " + str(distAdjust))

        #Probe up a quarter of touch plate second
        #once medium speed, once slow speed
        distAdjust = 0
        for feed, dist in zip([config.probing_settings.probe_feed_rate_fast, config.probing_settings.probe_feed_rate_slow], [firstSafeDist, firstSafeDist]):
            self.work_offset_move(x = math.cos(angle) * (dist - distAdjust) + math.cos(angle + math.pi/2.0) * plateWidth * 0.25 , \
                                  y = math.sin(angle) * (dist - distAdjust) + math.sin(angle + math.pi/2.0) * plateWidth * 0.25, feed=400)
            # Move below touch plate
            self.work_offset_move(z = plateHeight - 0.1, feed=100)
            # Probe to the touch plate
            ref2 = self.probe(x = math.cos(angle) * probeToDist + math.cos(angle + math.pi/2.0) * plateWidth * 0.25 , \
                              y = math.sin(angle) * probeToDist + math.sin(angle + math.pi/2.0) * plateWidth * 0.25, feed=feed)
            distAdjust = dist - math.dist([x,y], [ref2[0], ref2[1]]) - 0.05
        
        # Move away from reference plate and up
        self.work_offset_move(x = math.cos(angle) * firstSafeDist + math.cos(angle + math.pi/2.0) * plateWidth * 0.25 , \
                              y = math.sin(angle) * firstSafeDist + math.sin(angle + math.pi/2.0) * plateWidth * 0.25, feed=400)
        self.work_offset_move(z = plateHeight + 0.5, feed=180) # Move just above reference plate, clearing lip on reference plate

        yAxisAngle = math.atan2(ref2[1] - ref1[1], ref2[0] - ref1[0])
        xAxisAngle = yAxisAngle - math.pi / 2.0 % (2 * math.pi)
        print("estPlateAngle:" + str(estPlateAngle))
        print("xAxisAngle: " + str(xAxisAngle))
        return xAxisAngle

    def probeSequence(self, estPlateAngle, justZ):
        #function assumes spindle is directly above probe plate in estimated middle
        #function returns x, y, z position of center top of plate

        # First Zero out work coord offset with best we have thus far
        self.set_cur_pos_as(x=0, y=0, z = 0) # probe ony works on work coordinage system, set it to 0 so we know where we are in that
        # first get Z height right
        x, y, z = self.probeZSequence()
        if justZ:
            #reset G92 coordinate system to normal,
            #where current position is actual position is position in work coordinate system
            self.set_cur_pos_as(x = x, y = y, z = self.plateHeight + 0.5)
            self.set_work_coord_offset(x = 0.0, y = 0.0, Z = z - self.plateHeight)
            return [x, y, z]
        plateAngle = self.probeAngleOfTouchPlate(estPlateAngle, x, y)
        xy =  self.probeXYSequence(plateAngle)
        print("xy: " + str(xy))
        self.set_work_coord_offset(x, y, z)
        return xy + [z]
            
    def zero_on_refPlate(self, refPoints, justZ = False):
        avgX = (refPoints[0][0] + refPoints[1][0] + refPoints[2][0] + refPoints[3][0]) / 4.0
        avgY = (refPoints[0][1] + refPoints[1][1] + refPoints[2][1] + refPoints[3][1]) / 4.0
        angle = getBoxAngle(refPoints)


        self.flushGcodeRespQue()
        self.set_inches()
        self.absolute_move(None, None, -0.25, feed = 180) # Move close to Z limit
        # move 1.75" away from charuco marker bottom left
        self.absolute_move(avgX + 1.335*math.cos(math.pi*5/4), avgY + 1.335*math.sin(math.pi*5/4), None,  feed = 300) # Move above estimated ref plate

        print("avgXY: " + str(avgX) + " " + str(avgY))
        #first test out zero angle, then test out actual angle
        #return self.probeSequence(0)

        return self.probeSequence(angle, justZ)

    def waitOnGCodeComplete(self, gCode, timeout = 120.0, holdOnTimeout = True):
      # Block until a controller response containing gCode arrives.  Queue.get()
      # sleeps the calling thread (no busy-wait) and is woken by gerbil_callback.
      # The deadline bounds the total wait across non-matching responses; on
      # expiry a feed hold (!) is sent so the machine stops moving instead of
      # continuing while nothing is watching its responses.
      deadline = time.monotonic() + timeout
      resp = None
      while resp is None:
        remaining = deadline - time.monotonic()
        try:
          data = self.respQueue.get(timeout = max(remaining, 0))
        except queue.Empty:
          if holdOnTimeout:
            self.gerbil.hold()
          raise TimeoutError(
              "No '{}' response from controller within {}s{}".format(
                  gCode, timeout,
                  "; feed hold (!) sent - resume (~) or reset before continuing"
                  if holdOnTimeout else ""))
        print("    " + str(data))
        if gCode in str(data):
          resp = data
      print("resp: " + str(resp))
      print("Found: " + str(resp[0]))
      if isinstance(resp[0], dict):
          return resp[0][gCode]
      else:
          return resp[0]

    def flushGcodeRespQue(self):
        # Drain any stale responses left over from a previous operation.
        try:
            while True:
                self.respQueue.get_nowait()
        except queue.Empty:
            pass

    def is_busy(self):
        """True while a background CNC operation is in progress."""
        with self._job_lock:
            return self._job_thread is not None and self._job_thread.is_alive()

    def run_async(self, name, func, *args, **kwargs):
        """Run a blocking CNC operation on a background thread so the GUI does
        not freeze.  Only one operation may run at a time; calls made while an
        operation is already in progress are ignored (it is unsafe to interleave
        machine moves)."""
        with self._job_lock:
            if self._job_thread is not None and self._job_thread.is_alive():
                print("CNC busy; ignoring '{}'".format(name))
                return False

            def runner():
                try:
                    func(*args, **kwargs)
                except Exception:
                    print("Error in CNC operation '{}':".format(name))
                    traceback.print_exc()

            self._job_thread = threading.Thread(target=runner, name=name, daemon=True)
            self._job_thread.start()
            return True

    def _get_xyz_string(self, x = None, y = None, z = None):
        if x == None:
            xStr = ""
        else:
            xStr = " X" + str(x)

        if y == None:
            yStr = ""
        else:
            yStr = " Y" + str(y)

        if z == None:
            zStr = ""
        else:
            zStr = " Z" + str(z)
        return xStr, yStr, zStr

    def absolute_move(self, x = None, y = None , z = None, feed = 100):
        xStr, yStr, zStr = self._get_xyz_string(x, y, z)
        fStr = " F" + str(feed)
        self.set_inches()
        self.gerbil.send_immediately("G53 G1" + xStr + yStr + zStr + fStr + "\n")

    def work_offset_move(self, x = None, y = None , z = None, feed = 100):
        xStr, yStr, zStr = self._get_xyz_string(x, y, z)
        fStr = " F" + str(feed)
        self.gerbil.send_immediately("G1" + xStr + yStr + zStr + fStr + "\n")

    def send_svf(self, cncPaths, frame = None):
      global materialThickness
      global cutterDiameter

      cncGcodeGenerator = cncGcodeGeneratorClass(cncPaths           = cncPaths,
                                           materialThickness  = config.cutting_parameters.material_thickness,
                                           depthBelowMaterial = config.cutting_parameters.depth_below_material,
                                           depthPerPass       = config.cutting_parameters.depth_per_pass,
                                           cutFeedRate        = config.cutting_parameters.cut_feed_rate,
                                           safeHeight         = config.cutting_parameters.safe_height,
                                           tabHeight          = config.cutting_parameters.tab_height,
                                           useMM              = False # use inches
                                          )
      cncGcodeGenerator.Generate()
      cncGcodeGenerator.Save("test.nc")
      #asdfasdf
      self.set_inches()
      self.absolute_move(z = -0.5)
      print("SENDING GCODE")
      gCodeStrs = []
      for code in cncGcodeGenerator.gCodes:
          gCodeStrs.append(str(code))
      if frame is not None and frame.z.is_mesh:
          # Bend cut Z to follow the probed surface (constant depth of cut)
          gCodeStrs = warp_gcode_lines(gCodeStrs, frame.z.z_at,
                                       nominal = frame.z.nominal.value)
      # put whole file in buffer then run the job
      self.gerbil.write(gCodeStrs)
      self.gerbil.job_run()
      #for gCode in cncGcodeGenerator.gCodes:
      #    code = str(gCode) + "\n"
      #    #print("CODE:" + code)
      #    self.gerbil.stream(code)
      #    #self.gerbil.send_immediately(code)
      #    time.sleep(0.01)

      self.absolute_move(z = -0.25)

    def send_drawnPoints(self, offset, points3D, frame = None):
      global materialThickness
      global cutterDiameter
      points = deepcopy(points3D)
      for point in points:
          point.X = point.X + offset.X
          point.Y = point.Y + offset.Y
      cncPaths = cncPathsClass(points3D        = points,
                               pointsPerCurve  = 30,
                               distPerTab      = 8,
                               tabWidth        = 0.25,
                               cutterDiameter  = cutterDiameter
                        )
      cncGcodeGenerator = cncGcodeGeneratorClass(cncPaths           = cncPaths,
                                           materialThickness  = config.cutting_parameters.material_thickness,
                                           depthBelowMaterial = config.cutting_parameters.depth_below_material,
                                           depthPerPass       = config.cutting_parameters.depth_per_pass,
                                           cutFeedRate        = config.cutting_parameters.cut_feed_rate,
                                           safeHeight         = config.cutting_parameters.safe_height,
                                           tabHeight          = config.cutting_parameters.tab_height,
                                           useMM              = False # use inches
                                          )
      cncGcodeGenerator.Generate()
      self.set_inches()
      self.absolute_move(z = -0.25)
      print("SENDING GCODE")
      gCodeStrs = [str(gCode) for gCode in cncGcodeGenerator.gCodes]
      if frame is not None and frame.z.is_mesh:
          # Bend cut Z to follow the probed surface (constant depth of cut)
          gCodeStrs = warp_gcode_lines(gCodeStrs, frame.z.z_at,
                                       nominal = frame.z.nominal.value)
      for code in gCodeStrs:
          print("CODE: " + code)
          self.gerbil.stream(code + "\n")

      self.absolute_move(z = -0.25)

    def set_inches(self):
        self.gerbil.send_immediately("G20\n")

    def set_mm(self):
        self.gerbil.send_immediately("G21\n")

    def send_file(self, gCodeFile, xOffset, yOffset, rotation):
        #Set to inches for offset and rotations
        self.set_inches()

        ##########################################
        #Offset work to desired offset
        ##########################################
        self.set_work_coord_offset(xOffset, yOffset)

        ##########################################
        #Rotate work to desired rotation
        ##########################################
        deg = -rotation * 180 / math.pi
        self.gerbil.send_immediately("G68 X0 Y0 R" + str(deg) + "\n")

        #Set back to mm, typically the units g code assumes
        self.set_mm()

        ZFound = False
        with open(gCodeFile, 'r') as fh:
          for line_text in fh.readlines():
            if " Z" in line_text.upper():
              ZFound = True
              break

        # if Z move found in file (not a laser cutting file), then move cutter away from workspace as first move
        # so that if it was forgotten to do that, the first rapid traverse does not run into the workpiece
        if ZFound:
          self.absolute_move(z = -0.25)

        with open(gCodeFile, 'r') as fh:
            for line_text in fh.readlines():
                self.gerbil.stream(line_text)

        # Turn off rotated coordinate system
        self.gerbil.send_immediately("G69\n")


class GCodeSenderMachine:
    """Adapts GCodeSender to the probing.base.Machine protocol so probe
    strategies (probing/strategies.py) can drive it without knowing about
    gerbil. The few lines PROBING_DESIGN.md promised."""

    def __init__(self, sender):
        self.sender = sender

    def probe(self, x = None, y = None, z = None, feed = 5.9):
        return self.sender.probe(x, y, z, feed)

    def move(self, x = None, y = None, z = None, feed = 100):
        self.sender.work_offset_move(x, y, z, feed)

    def position(self):
        return self.sender.get_absolute_pos()


#############################################################################
# Startup pipeline
#
# Behavior-preserving extraction of what used to run at module import:
# capture -> calibrate -> preview -> overlay -> plt.show().  Split into
# functions so another shell (the planned PySide6 UI) can import this module
# and call the pieces -- especially calibrate_bed() on a saved image --
# without opening windows or touching hardware.  Running this file directly
# behaves exactly as before via main().
#############################################################################

def capture_bed_image(useCamera = False):
    """Open the capture device and grab a frame of the bed.  Live capture is
    disabled by default (matching prior behavior): the saved test image is
    used instead, without touching the camera.  Returns (cap, frame); cap is
    None in test-image mode."""
    if not useCamera:
        return None, cv2.imread('cnc13.jpg')
    vision = config.vision_settings
    cap = cv2.VideoCapture(vision.camera_device_index, cv2.CAP_DSHOW) # Set Capture Device
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, vision.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, vision.camera_height)
    ret, frame = cap.read()
    if frame is not None:
        from camera_intrinsics import to_scene_frame
        frame = to_scene_frame(frame, vision.camera_device_index,
                               vision.camera_rotation)
    return cap, frame


def calibrate_bed(frame):
    """Detect the ArUco markers on the vertical rails and compute the
    bed<->image homographies.  Annotates `frame` with the detections (marker
    arrows, rail boxes, bed outline) exactly as before -- the annotated frame
    is what gets warped into the overhead bed view.  No windows, no hardware:
    callable on any saved image.

    Returns (annotatedFrame, bedPixelToOrigPixelLoc, origPixelToBedPixelLoc,
    boxes, ids, info).  info carries the detection details for UIs:
    {"mode", "counts", "plate_found", and "rms" on the machine-tags path}."""
    #######################################################################
    # Preferred path: flat AprilTag strips on the rail tops / bed, solved
    # via PnP with the lens intrinsics.  Falls through to the legacy
    # vertical-rail ChArUco path when no machine tags are in view.
    #######################################################################
    import machine_tags
    machine_result = machine_tags.try_calibrate(frame, config)
    if machine_result is not None:
        ids = machine_result["ids"]
        info = {"mode": "machine_tags", "counts": machine_result["counts"],
                "rms": machine_result["rms"], "plate_found": bool(66 in ids)}
        return (machine_result["frame"], machine_result["bed_to_orig"],
                machine_result["orig_to_bed"], machine_result["boxes"],
                ids, info)

    ########################################
    # Get aruco box information
    ########################################
    from camera_intrinsics import detect_aruco
    boxes, ids = detect_aruco(frame, cv2.aruco.DICT_4X4_100)
    if ids is None or len(ids) == 0:
        raise RuntimeError("no ArUco markers visible in the image - "
                           "check that the camera can see both rail marker strips")
    # OpenCV 4 returns Nx1 while newer builds may return a flat vector.
    # The legacy geometry helpers consume the Nx1 shape.
    ids = np.asarray(ids, dtype=np.int32).reshape(-1, 1)

    pixelLoc = [None]*2
    locations = [None]*2
    sideRefLocToOrigPixelLoc = [None]*2
    pixelsAtBed = [None]*2
    refBoxes = [leftBoxRef, rightBoxRef]
    ########################################
    # Determine vertical homography at left (i=0) and right (i=1) side of CNC machine
    ########################################
    for i in range(0, 2):
      pixelLoc[i],  locations[i],  frame = boxes_to_point_and_location_list(boxes, ids, frame, i == 1)
      if len(pixelLoc[i]) < 4:
        raise RuntimeError("%s rail markers not visible (%d of %d detected markers "
                           "belong to that rail)"
                           % ("right" if i == 1 else "left", len(pixelLoc[i]) // 4, len(ids)))
      print(ids)
      for location in locations[i]:
        print(location)

      ########################################
      #Determine forward and backward transformation through homography
      ########################################
      sideRefLocToOrigPixelLoc[i], status = cv2.findHomography(np.array(locations[i]), np.array(pixelLoc[i]))

      #############################################################
      # Draw vertical box on left and right vertical region of CNC
      #############################################################
      points = np.array([[refBoxes[i].Z,0],[bedSize.Z,0],[bedSize.Z,bedSize.Y],[refBoxes[i].Z,bedSize.Y]])
      pixelsAtBed[i] = cv2.perspectiveTransform(points.reshape(-1,1,2), sideRefLocToOrigPixelLoc[i])
      display_4_lines(pixelsAtBed[i], frame)

    ####################################################################################################
    # Get forward and backward homography from simulated overhead Pixel location to Orig pixel location
    # Makes destination image same size as source image.  Reshaped later due to matplot lib speed limitations
    ####################################################################################################
    #shape[0] is height.  shape[1] is width
    #PixelCorners are [height,0], height, width
    height = float(frame.shape[1])
    width  = float(frame.shape[0])
    bedPixelCorners = np.array([[height,0.0],[height,width],[0.0,0.0],[0.0,width]])
    refPixels = np.array([pixelsAtBed[0][1],pixelsAtBed[0][2],pixelsAtBed[1][1],pixelsAtBed[1][2]])
    bedPixelToOrigPixelLoc, status    = cv2.findHomography(bedPixelCorners, refPixels)
    origPixelToBedPixelLoc, status    = cv2.findHomography(refPixels, bedPixelCorners)

    #############################################################
    # Draw box on CNC bed
    #############################################################
    pixels = cv2.perspectiveTransform(bedPixelCorners.reshape(-1,1,2), bedPixelToOrigPixelLoc)
    display_4_lines(pixels, frame, flip=True)

    info = {"mode": "legacy",
            "counts": {"Left rail": len(pixelLoc[0]) // 4,
                       "Right rail": len(pixelLoc[1]) // 4},
            "plate_found": bool(66 in ids)}
    return frame, bedPixelToOrigPixelLoc, origPixelToBedPixelLoc, boxes, ids, info


def warp_to_overhead(frame, origPixelToBedPixelLoc):
    """Warp the (annotated) camera frame to the square overhead bed view that
    the overlay UI displays."""
    cv2Overhead = cv2.warpPerspective(frame, origPixelToBedPixelLoc, (frame.shape[1], frame.shape[0]))
    return cv2.resize(cv2Overhead, (bedViewSizePixels, bedViewSizePixels))


def locate_touch_plate(frame, boxes, ids, origPixelToBedPixelLoc, markerId = 66):
    """Find the probe touch plate marker in the image and return its corner
    pixel locations in the overhead bed view, ready for
    OverlayGcode.set_ref_loc()."""
    refPixelLoc    = get_id_loc(frame, boxes, ids, markerId)
    if refPixelLoc is None:
        print("Touch plate marker %d not visible" % markerId)
        return []
    refPhysicalLoc = cv2.perspectiveTransform(refPixelLoc.reshape(-1,1,2), origPixelToBedPixelLoc)
    touchPlateLocPercent = refPhysicalLoc / [frame.shape[1], frame.shape[0]]
    touchPlateLoc = []
    touchPlatePixels = []
    for a in touchPlateLocPercent:
        touchPlateLoc.append(a[0] * [bedSize.X, bedSize.Y])
        touchPlatePixels.append(a[0] * [bedViewSizePixels, bedViewSizePixels] )
    print("Touch Plate Loc: " + str(touchPlateLoc))
    return touchPlatePixels


def main(svgFile = 'puzzles2.svg', gCodeFile = None, enableSender = False,
         useCamera = False, showCalibration = True):
    """Run the matplotlib application: capture, calibrate, show the
    calibration preview (blocks until a key is pressed in the OpenCV window),
    then the interactive overlay until its window closes."""
    cap, frame = capture_bed_image(useCamera)
    frame, bedPixelToOrigPixelLoc, origPixelToBedPixelLoc, boxes, ids, _info = calibrate_bed(frame)

    if showCalibration:
        #############################################################
        # Display bed on original image
        #############################################################
        preview = cv2.resize(frame, (1280, 700))
        cv2.imshow('image', preview)
        cv2.waitKey()

    ######################################################################
    # Warp perspective to perpendicular to bed view, create overlay class
    ######################################################################
    overlay = OverlayGcode(warp_to_overhead(frame, origPixelToBedPixelLoc),
                           svgFile = svgFile, gCodeFile = gCodeFile,
                           enableSender = enableSender)
    overlay.set_ref_loc(locate_touch_plate(frame, boxes, ids, origPixelToBedPixelLoc))

    plt.show()

    # When everything done, release the capture
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    return overlay


if __name__ == "__main__":
    main()
