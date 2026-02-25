import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw
from scipy.ndimage import center_of_mass, shift
from skimage.transform import resize
from skimage.feature import hog
import joblib
from pathlib import Path

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw number - Guess Number")
        
        # Load model và scaler
        # self.model = joblib.load("./svm_hog_model.joblib")
        # self.scaler = joblib.load("./scaler.joblib")
        MODEL_DIR = Path(r"C:\Users\vuong\OneDrive\Desktop\Ml2\ML2-Final\SVM")
        self.model = joblib.load(MODEL_DIR / "svm_hog_model.joblib")
        self.scaler = joblib.load(MODEL_DIR / "scaler.joblib")
        
        # Thiết lập canvas
        self.canvas_width = 600
        self.canvas_height = 400
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, 
                                bg='black', cursor='cross')
        self.canvas.pack(pady=10)
        
        # PIL Image để vẽ
        self.image = Image.new('L', (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)
        
        # Biến theo dõi chuột
        self.last_x = None
        self.last_y = None
        
        # Bind sự kiện chuột
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset_position)
        
        # Frame cho các nút
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)
        
        # Nút Dự đoán
        predict_btn = tk.Button(button_frame, text="Predict", command=self.predict,
                               bg='green', fg='white', font=('Arial', 12, 'bold'),
                               width=12, height=2)
        predict_btn.pack(side=tk.LEFT, padx=5)
        
        # Nút Xóa
        clear_btn = tk.Button(button_frame, text="Clear", command=self.clear_canvas,
                             bg='red', fg='white', font=('Arial', 12, 'bold'),
                             width=12, height=2)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Label kết quả
        self.result_label = tk.Label(root, text="Draw a number in range 0-9",
                                    font=('Arial', 14))
        self.result_label.pack(pady=10)
        
        # Label hướng dẫn
    
    def start_draw(self, event):
        self.last_x = event.x
        self.last_y = event.y
    
    def paint(self, event):
        if self.last_x and self.last_y:
            x, y = event.x, event.y
            # Vẽ trên canvas
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                   fill='white', width=20, capstyle=tk.ROUND,
                                   smooth=tk.TRUE)
            # Vẽ trên PIL image
            self.draw.line([self.last_x, self.last_y, x, y], fill=255, width=20)
            self.last_x = x
            self.last_y = y
    
    def reset_position(self, event):
        self.last_x = None
        self.last_y = None
    
    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Draw a number in range 0-9")
    
    def preprocess_image(self, img_array):
        # Normalize 0-1
        img = img_array.astype(np.float32) / 255.0
        
        # Nếu canvas trống
        if img.max() == 0:
            return None
        
        # Invert để digit là trắng trên nền đen
        # (canvas đã là trắng trên đen rồi, không cần invert)
        
        # Tìm bounding box
        coords = np.column_stack(np.where(img > 0))
        if len(coords) == 0:
            return None
            
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        digit = img[y_min:y_max+1, x_min:x_max+1]
        
        # Resize to 20x20
        digit = resize(digit, (20, 20), anti_aliasing=True)
        
        # Đặt vào canvas 28x28
        canvas = np.zeros((28, 28))
        canvas[4:24, 4:24] = digit
        
        # Center of mass shift
        cy, cx = center_of_mass(canvas)
        shift_y = 14 - cy
        shift_x = 14 - cx
        canvas = shift(canvas, (shift_y, shift_x))
        
        return canvas
    
    def predict(self):
        # Lấy image từ PIL
        img_array = np.array(self.image)
        
        # Preprocess
        processed = self.preprocess_image(img_array)
        
        if processed is None:
            self.result_label.config(text="Please draw a number!", fg='red')
            return
        
        # Extract HOG features
        features = hog(
            processed,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        hog_features = features.reshape(1, -1)
        hog_scaled = self.scaler.transform(hog_features)
        
        # Predict
        prediction = self.model.predict(hog_scaled)
        confidence = self.model.decision_function(hog_scaled)
        max_confidence = np.max(confidence)
        
        self.result_label.config(
            text=f"Predicted number: {prediction[0]} (Confidence: {max_confidence:.2f})",
            fg='blue',
            font=('Arial', 16, 'bold')
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
