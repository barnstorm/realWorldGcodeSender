"""Camera discovery for the calibration view.

QtMultimedia lists video inputs by name without opening them, so it is
instant; the OpenCV probe in utils.device_detection is the fallback and
opens every index, which can take seconds.  Both report devices in system
enumeration order -- the same order cv2.CAP_DSHOW resolves indices in
capture_bed_image().
"""


def available_cameras():
    """Return [(device_index, label), ...] for every attached video input."""
    try:
        from PySide6.QtMultimedia import QMediaDevices
        return [(index, device.description() or ("Camera %d" % index))
                for index, device in enumerate(QMediaDevices.videoInputs())]
    except ImportError:
        from utils.device_detection import DeviceDetector
        return [(cam.index, cam.name) for cam in DeviceDetector.get_cameras()]
