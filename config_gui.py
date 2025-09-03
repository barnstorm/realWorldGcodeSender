"""
Configuration GUI for realWorldGcodeSender

Provides a user-friendly interface for editing configuration settings.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
from app_config import Configuration, get_config, save_config


class ConfigurationGUI:
    """GUI for editing configuration settings"""
    
    def __init__(self, root=None):
        if root is None:
            self.root = tk.Tk()
            self.root.title("realWorldGcodeSender Configuration")
            self.root.geometry("800x700")
            self.own_root = True
        else:
            self.root = root
            self.own_root = False
        
        self.config = get_config()
        
        # Initialize all section variable dictionaries
        self.physical_vars = {}
        self.cutting_vars = {}
        self.vision_vars = {}
        self.comm_vars = {}
        self.probing_vars = {}
        
        self.create_widgets()
        self.load_values()
    
    def create_widgets(self):
        """Create GUI widgets"""
        # Create main frame with scrollbar
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create canvas and scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Physical Setup Section
        self.create_physical_setup_section(scrollable_frame)
        
        # Cutting Parameters Section
        self.create_cutting_parameters_section(scrollable_frame)
        
        # Vision Settings Section
        self.create_vision_settings_section(scrollable_frame)
        
        # Communication Settings Section  
        self.create_communication_settings_section(scrollable_frame)
        
        # Probing Settings Section
        self.create_probing_settings_section(scrollable_frame)
        
        # Buttons
        self.create_buttons(scrollable_frame)
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    def create_physical_setup_section(self, parent):
        """Create physical setup configuration section"""
        frame = ttk.LabelFrame(parent, text="Physical Setup", padding=10)
        frame.pack(fill="x", pady=5)
        
        # Box width
        self.create_float_entry(frame, "box_width", "ChArUco Box Width (inches):", 0, 0, self.physical_vars)
        
        # Bed dimensions
        self.create_float_entry(frame, "bed_size_x", "Bed Size X (inches):", 1, 0, self.physical_vars)
        self.create_float_entry(frame, "bed_size_y", "Bed Size Y (inches):", 2, 0, self.physical_vars)
        self.create_float_entry(frame, "bed_size_z", "Bed Size Z (inches):", 3, 0, self.physical_vars)
        
        # Right reference box
        ttk.Label(frame, text="Right Reference Box:").grid(row=4, column=0, columnspan=2, sticky="w", pady=(10,0))
        self.create_float_entry(frame, "right_box_ref_x", "  X Position:", 5, 0, self.physical_vars)
        self.create_float_entry(frame, "right_box_ref_y", "  Y Position:", 6, 0, self.physical_vars)
        self.create_float_entry(frame, "right_box_ref_z_offset", "  Z Offset from Bed:", 7, 0, self.physical_vars)
        self.create_float_entry(frame, "right_box_far_height", "  Far End Height:", 8, 0, self.physical_vars)
        
        # Left reference box
        ttk.Label(frame, text="Left Reference Box:").grid(row=9, column=0, columnspan=2, sticky="w", pady=(10,0))
        self.create_float_entry(frame, "left_box_ref_x", "  X Position:", 10, 0, self.physical_vars)
        self.create_float_entry(frame, "left_box_ref_y", "  Y Position:", 11, 0, self.physical_vars)
        self.create_float_entry(frame, "left_box_ref_z_offset", "  Z Offset from Bed:", 12, 0, self.physical_vars)
        self.create_float_entry(frame, "left_box_far_height", "  Far End Height:", 13, 0, self.physical_vars)
    
    def create_cutting_parameters_section(self, parent):
        """Create cutting parameters configuration section"""
        frame = ttk.LabelFrame(parent, text="Cutting Parameters", padding=10)
        frame.pack(fill="x", pady=5)
        
        self.create_float_entry(frame, "material_thickness", "Material Thickness (inches):", 0, 0, self.cutting_vars)
        self.create_float_entry(frame, "cutter_diameter", "Cutter Diameter (inches):", 1, 0, self.cutting_vars)
        self.create_float_entry(frame, "cut_feed_rate", "Cut Feed Rate:", 2, 0, self.cutting_vars)
        self.create_float_entry(frame, "depth_per_pass", "Depth per Pass (inches):", 3, 0, self.cutting_vars)
        self.create_float_entry(frame, "depth_below_material", "Depth Below Material (inches):", 4, 0, self.cutting_vars)
        self.create_float_entry(frame, "safe_height", "Safe Height (inches):", 5, 0, self.cutting_vars)
        self.create_float_entry(frame, "tab_height", "Tab Height (inches):", 6, 0, self.cutting_vars)
    
    def create_vision_settings_section(self, parent):
        """Create vision settings configuration section"""
        frame = ttk.LabelFrame(parent, text="Vision Settings", padding=10)
        frame.pack(fill="x", pady=5)
        
        self.create_int_entry(frame, "bed_view_size_pixels", "Bed View Size (pixels):", 0, 0, self.vision_vars)
        self.create_int_entry(frame, "camera_device_index", "Camera Device Index:", 1, 0, self.vision_vars)
        self.create_int_entry(frame, "camera_width", "Camera Width:", 2, 0, self.vision_vars)
        self.create_int_entry(frame, "camera_height", "Camera Height:", 3, 0, self.vision_vars)
    
    def create_communication_settings_section(self, parent):
        """Create communication settings configuration section"""
        frame = ttk.LabelFrame(parent, text="Communication Settings", padding=10)
        frame.pack(fill="x", pady=5)
        
        self.create_string_entry(frame, "com_port", "COM Port:", 0, 0, self.comm_vars)
        self.create_int_entry(frame, "baud_rate", "Baud Rate:", 1, 0, self.comm_vars)
        self.create_bool_entry(frame, "auto_detect_port", "Auto-detect Port:", 2, 0, self.comm_vars)
    
    def create_probing_settings_section(self, parent):
        """Create probing settings configuration section"""
        frame = ttk.LabelFrame(parent, text="Probing Settings", padding=10)
        frame.pack(fill="x", pady=5)
        
        self.create_float_entry(frame, "plate_height", "Touch Plate Height (inches):", 0, 0, self.probing_vars)
        self.create_float_entry(frame, "plate_width", "Touch Plate Width (inches):", 1, 0, self.probing_vars)
        self.create_float_entry(frame, "probe_feed_rate_fast", "Probe Feed Rate (Fast):", 2, 0, self.probing_vars)
        self.create_float_entry(frame, "probe_feed_rate_slow", "Probe Feed Rate (Slow):", 3, 0, self.probing_vars)
        self.create_float_entry(frame, "dist_to_notch", "Distance to Notch (inches):", 4, 0, self.probing_vars)
    
    def create_float_entry(self, parent, var_name, label_text, row, col, section_vars):
        """Create a float input entry"""
        ttk.Label(parent, text=label_text).grid(row=row, column=col*2, sticky="w", padx=(0,5))
        var = tk.StringVar()
        entry = ttk.Entry(parent, textvariable=var, width=20)
        entry.grid(row=row, column=col*2+1, sticky="w", padx=(0,10))
        section_vars[var_name] = var
    
    def create_int_entry(self, parent, var_name, label_text, row, col, section_vars):
        """Create an integer input entry"""
        ttk.Label(parent, text=label_text).grid(row=row, column=col*2, sticky="w", padx=(0,5))
        var = tk.StringVar()
        entry = ttk.Entry(parent, textvariable=var, width=20)
        entry.grid(row=row, column=col*2+1, sticky="w", padx=(0,10))
        section_vars[var_name] = var
    
    def create_string_entry(self, parent, var_name, label_text, row, col, section_vars):
        """Create a string input entry"""
        ttk.Label(parent, text=label_text).grid(row=row, column=col*2, sticky="w", padx=(0,5))
        var = tk.StringVar()
        entry = ttk.Entry(parent, textvariable=var, width=20)
        entry.grid(row=row, column=col*2+1, sticky="w", padx=(0,10))
        section_vars[var_name] = var
    
    def create_bool_entry(self, parent, var_name, label_text, row, col, section_vars):
        """Create a boolean input entry"""
        var = tk.BooleanVar()
        check = ttk.Checkbutton(parent, text=label_text, variable=var)
        check.grid(row=row, column=col*2, columnspan=2, sticky="w", padx=(0,10))
        section_vars[var_name] = var
    
    def create_buttons(self, parent):
        """Create control buttons"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill="x", pady=20)
        
        ttk.Button(button_frame, text="Load Default", command=self.load_default).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Load from File", command=self.load_from_file).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Save", command=self.save_config).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Save As", command=self.save_as).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Apply", command=self.apply_config).pack(side="right", padx=5)
        
        if self.own_root:
            ttk.Button(button_frame, text="Close", command=self.root.destroy).pack(side="right", padx=5)
    
    def load_values(self):
        """Load current configuration values into GUI"""
        # Physical setup
        ps = self.config.physical_setup
        self.physical_vars["box_width"].set(str(ps.box_width))
        self.physical_vars["bed_size_x"].set(str(ps.bed_size_x))
        self.physical_vars["bed_size_y"].set(str(ps.bed_size_y))
        self.physical_vars["bed_size_z"].set(str(ps.bed_size_z))
        self.physical_vars["right_box_ref_x"].set(str(ps.right_box_ref_x))
        self.physical_vars["right_box_ref_y"].set(str(ps.right_box_ref_y))
        self.physical_vars["right_box_ref_z_offset"].set(str(ps.right_box_ref_z_offset))
        self.physical_vars["right_box_far_height"].set(str(ps.right_box_far_height))
        self.physical_vars["left_box_ref_x"].set(str(ps.left_box_ref_x))
        self.physical_vars["left_box_ref_y"].set(str(ps.left_box_ref_y))
        self.physical_vars["left_box_ref_z_offset"].set(str(ps.left_box_ref_z_offset))
        self.physical_vars["left_box_far_height"].set(str(ps.left_box_far_height))
        
        # Cutting parameters
        cp = self.config.cutting_parameters
        self.cutting_vars["material_thickness"].set(str(cp.material_thickness))
        self.cutting_vars["cutter_diameter"].set(str(cp.cutter_diameter))
        self.cutting_vars["cut_feed_rate"].set(str(cp.cut_feed_rate))
        self.cutting_vars["depth_per_pass"].set(str(cp.depth_per_pass))
        self.cutting_vars["depth_below_material"].set(str(cp.depth_below_material))
        self.cutting_vars["safe_height"].set(str(cp.safe_height))
        self.cutting_vars["tab_height"].set(str(cp.tab_height))
        
        # Vision settings
        vs = self.config.vision_settings
        self.vision_vars["bed_view_size_pixels"].set(str(vs.bed_view_size_pixels))
        self.vision_vars["camera_device_index"].set(str(vs.camera_device_index))
        self.vision_vars["camera_width"].set(str(vs.camera_width))
        self.vision_vars["camera_height"].set(str(vs.camera_height))
        
        # Communication settings
        cs = self.config.communication_settings
        self.comm_vars["com_port"].set(str(cs.com_port))
        self.comm_vars["baud_rate"].set(str(cs.baud_rate))
        self.comm_vars["auto_detect_port"].set(cs.auto_detect_port)
        
        # Probing settings
        ps = self.config.probing_settings
        self.probing_vars["plate_height"].set(str(ps.plate_height))
        self.probing_vars["plate_width"].set(str(ps.plate_width))
        self.probing_vars["probe_feed_rate_fast"].set(str(ps.probe_feed_rate_fast))
        self.probing_vars["probe_feed_rate_slow"].set(str(ps.probe_feed_rate_slow))
        self.probing_vars["dist_to_notch"].set(str(ps.dist_to_notch))
    
    def apply_config(self):
        """Apply GUI values to configuration"""
        try:
            # Update physical setup
            ps = self.config.physical_setup
            ps.box_width = float(self.physical_vars["box_width"].get())
            ps.bed_size_x = float(self.physical_vars["bed_size_x"].get())
            ps.bed_size_y = float(self.physical_vars["bed_size_y"].get())
            ps.bed_size_z = float(self.physical_vars["bed_size_z"].get())
            ps.right_box_ref_x = float(self.physical_vars["right_box_ref_x"].get())
            ps.right_box_ref_y = float(self.physical_vars["right_box_ref_y"].get())
            ps.right_box_ref_z_offset = float(self.physical_vars["right_box_ref_z_offset"].get())
            ps.right_box_far_height = float(self.physical_vars["right_box_far_height"].get())
            ps.left_box_ref_x = float(self.physical_vars["left_box_ref_x"].get())
            ps.left_box_ref_y = float(self.physical_vars["left_box_ref_y"].get())
            ps.left_box_ref_z_offset = float(self.physical_vars["left_box_ref_z_offset"].get())
            ps.left_box_far_height = float(self.physical_vars["left_box_far_height"].get())
            
            # Update cutting parameters
            cp = self.config.cutting_parameters
            cp.material_thickness = float(self.cutting_vars["material_thickness"].get())
            cp.cutter_diameter = float(self.cutting_vars["cutter_diameter"].get())
            cp.cut_feed_rate = float(self.cutting_vars["cut_feed_rate"].get())
            cp.depth_per_pass = float(self.cutting_vars["depth_per_pass"].get())
            cp.depth_below_material = float(self.cutting_vars["depth_below_material"].get())
            cp.safe_height = float(self.cutting_vars["safe_height"].get())
            cp.tab_height = float(self.cutting_vars["tab_height"].get())
            
            # Update vision settings
            vs = self.config.vision_settings
            vs.bed_view_size_pixels = int(self.vision_vars["bed_view_size_pixels"].get())
            vs.camera_device_index = int(self.vision_vars["camera_device_index"].get())
            vs.camera_width = int(self.vision_vars["camera_width"].get())
            vs.camera_height = int(self.vision_vars["camera_height"].get())
            
            # Update communication settings
            cs = self.config.communication_settings
            cs.com_port = self.comm_vars["com_port"].get()
            cs.baud_rate = int(self.comm_vars["baud_rate"].get())
            cs.auto_detect_port = self.comm_vars["auto_detect_port"].get()
            
            # Update probing settings
            ps = self.config.probing_settings
            ps.plate_height = float(self.probing_vars["plate_height"].get())
            ps.plate_width = float(self.probing_vars["plate_width"].get())
            ps.probe_feed_rate_fast = float(self.probing_vars["probe_feed_rate_fast"].get())
            ps.probe_feed_rate_slow = float(self.probing_vars["probe_feed_rate_slow"].get())
            ps.dist_to_notch = float(self.probing_vars["dist_to_notch"].get())
            
            messagebox.showinfo("Success", "Configuration applied successfully")
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid value: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply configuration: {e}")
    
    def save_config(self):
        """Save current configuration"""
        self.apply_config()
        save_config()
        messagebox.showinfo("Success", "Configuration saved to config.json")
    
    def save_as(self):
        """Save configuration to a different file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.apply_config()
            self.config.save_to_file(filename)
            messagebox.showinfo("Success", f"Configuration saved to {filename}")
    
    def load_from_file(self):
        """Load configuration from file"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.config = Configuration.load_from_file(filename)
                self.load_values()
                messagebox.showinfo("Success", f"Configuration loaded from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")
    
    def load_default(self):
        """Load default configuration"""
        self.config = Configuration.get_default()
        self.load_values()
        messagebox.showinfo("Success", "Default configuration loaded")
    
    def run(self):
        """Run the GUI main loop"""
        if self.own_root:
            self.root.mainloop()


def main():
    """Main entry point for standalone execution"""
    app = ConfigurationGUI()
    app.run()


if __name__ == "__main__":
    main()