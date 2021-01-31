import cv2

def IOU(img:list):
    
    bz = img[0]
    ch = img[1]
    
    area_bz = cv2.bitwise_not(bz)
    area_ch = cv2.bitwise_not(ch)
    area_bzUch = cv2.bitwise_or(area_bz,area_ch)
    
    area_bz = area_bz[area_bz==255]
    area_ch = area_ch[area_ch==255]
    area_bzUch = area_bzUch[area_bzUch==255]
    
    area_bzIch = area_bz+area_ch-area_bzUch
    
    IOU = area_bzIch/area_bzUch
    
    bz_asymm = area_bzUch-area_ch
    ch_asymm = area_bzUch-area_bz
    
    return (IOU,bz_asymm,ch_asymm)
