import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2


white = (255,255,255)
black = (0,0,0)
red = (255, 0, 0)
# green = (0, 128, 0)

window_sizeX = 640
window_sizeY = 480
BoundaryInc = 5

save_image = False
img_count = 1

MODEL = load_model("bestmodel.h5")
Predict = True
LABLES = {0: "Zero", 1: "One",
        2: "Two", 3: "Three",
        4: "Four", 5: "Five", 
        6: "Six", 7: "Seven",
        8: "Eight", 9: "Nine"} 

# Initialize our pygame
pygame.init()

FONT = pygame.font.Font("freesansbold.ttf", 18)
display_surf = pygame.display.set_mode((window_sizeX, window_sizeY))
white_int = display_surf.map_rgb(white)
pygame.display.set_caption("Digit Board")

is_writing = False
number_Xcord = []
number_Ycord = []

while True:

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        
        if event.type == MOUSEMOTION and is_writing:
            Xcord, Ycord = event.pos
            pygame.draw.circle(display_surf, white, (Xcord, Ycord), 4, 0)

            number_Xcord.append(Xcord)
            number_Ycord.append(Ycord)

        if event.type == MOUSEBUTTONDOWN:
            is_writing = True
        
        if event.type == MOUSEBUTTONUP:
            is_writing = False
            number_Xcord = sorted(number_Xcord)
            number_Ycord = sorted(number_Ycord)

            rect_minX, rect_maxX = max(number_Xcord[0]-BoundaryInc, 0), min(window_sizeX, number_Xcord[-1]+BoundaryInc)
            rect_minY, rect_maxY = max(number_Ycord[0]-BoundaryInc, 0), min(window_sizeY, number_Ycord[-1]+BoundaryInc)

            number_Xcord = []
            number_Ycord = []
            
            img_arr = np.array(pygame.PixelArray(display_surf))[rect_minX:rect_maxX, rect_minY:rect_maxY].T.astype(np.float32)

            if save_image:
                cv2.imwrite("image.png")
                img_count += 1

            if Predict:
                image = cv2.resize(img_arr, (28,28))
                image = np.pad(image, (10,10), 'constant', constant_values = 0)
                image = cv2.resize(image, (28,28))/white_int

                lable = str(LABLES[np.argmax(MODEL.predict(image.reshape(1,28,28,1)))])
                
                textSurface = FONT.render(lable, True, red, white)
                textRectObj = textSurface.get_rect()
                textRectObj.left, textRectObj.bottom = rect_minX, rect_minY
                
                display_surf.blit(textSurface, textRectObj)
            
            if event.type == KEYDOWN:
                if event.unicode == 'n':
                    display_surf.fill(black)

        pygame.display.update()