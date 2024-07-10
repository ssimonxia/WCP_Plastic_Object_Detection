
class Config():
    def __init__(self):
        # The file path of the input video
        self.source = "sources/GX020093.mp4"
        
        # The file path of the model to use
        self.model = 'models/best_by_ayyan.pt'
        
        # The width and height of output video
        self.output_video_width = 1024
        self.output_video_height = 1024
        
        # The classes that the model classifies
        self.class_list = ["Aluminum", "Books", "Cardboard", "Glass", "HDPE", "LDPE", "OP", "PET", "PP", "PS", "RP"]
        
        # Whether to show the result in real time
        self.visualize = False
        
        # Position of the counting line
        self.line = [(0, 700), (1200, 700)]