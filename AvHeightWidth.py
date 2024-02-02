import os 
import cv2
import statistics as s

class ImageDirect:
# Specs for datasets 
    def __init__(self, path):
        self.path = path
        self.total_height = []
        self.total_width = []
        self.num_images = 0
        self.min_height = 10000000
        self.min_width = 10000000
    
    #Returns the average and minimum dimensions of the dataset input images
    def dim_finder(self):
        for root, dirs, files in os.walk(self.path):
            for subdirs in dirs:
                newdir = root+subdirs
                print(newdir)
                for r2, d2, f2 in os.walk(newdir):
                    for filename in f2:
                        if filename.endswith('.jpg'): 
                            img_path = os.path.join(r2, filename)
                            img = cv2.imread(img_path)
                            height, width, _ = img.shape
                            self.total_height.append(height)
                            self.total_width.append(width)
                            self.num_images += 1
                            if(height <= self.min_height):
                                self.min_height = height
                            if(width <= self.min_width):
                                self.min_width = width

        avg_height = s.mean(self.total_height)
        avg_width = s.mean(self.total_width)
        std_err_h = s.stdev(self.total_height)/(self.num_images**0.5)
        std_err_w = s.stdev(self.total_width)/(self.num_images**0.5)

        return (avg_height, avg_width, self.min_height, self.min_width, self.num_images, std_err_h, std_err_w)

if __name__ == "__main__":
    NS = "./NSdataset/"
    LH = "./LHdataset/"
    EM = "./EMdataset/"
    ND = "./NDdataset/"
    LHAug ='./LHdatasetAugX12/'

    image_dir = ImageDirect(LHAug)
    avg_height, avg_width, min_height, min_width, num_images, stderrh, stderrw = image_dir.dim_finder()
    #min_height, min_width = image_dir.min_dim()

    print("Average Dimensions:")
    print(f"Average height: {avg_height}" + " +- " + str(stderrh))
    print(f"Average width: {avg_width}" + " +- " + str(stderrw))
    print("Minimum Dimensions:")
    print(f"Minimum height: {min_height}")
    print(f"Minimum width: {min_width}\n\n")
    print(f"Number of Images: {num_images}" )


    




