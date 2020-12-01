import cv2

def de_noise(noisy_img):
    #-CV2 crazy denoiser magic-
    return cv2.fastNlMeansDenoising(noisy_img, h=30)
    #-Gaussian Blur-
    # sigmaX = (noisy_img.shape[0])/80
    # sigmaY = (noisy_img.shape[1])/80
    # return cv2.GaussianBlur(noisy_img, ksize=(0,0), sigmaX=sigmaX, sigmaY=sigmaY)
    #-Median Blur-
    #return cv2.medianBlur(noisy_img, 7)
