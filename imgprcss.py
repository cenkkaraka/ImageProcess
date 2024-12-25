import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
def calculate_clip_limit(hist,clip_pixels):
    clipped = np.copy(hist)
    while(clip_pixels > 0 ):
        sorted_indices = np.argsort(hist)[::-1]
        max_value = np.max(clipped)
        max_bin_indices = np.where(clipped == max_value)[0]
        second_max_bin_index = sorted_indices[len(max_bin_indices)]
        second_max_value = hist[second_max_bin_index]
        gap = max_value - second_max_value
        for i in max_bin_indices:
            clipped[i] -= gap
            clip_pixels -= gap
    clip_limit = clipped.max()
    return clip_limit ,clipped

def clip_histogram(hist,clip_limit):

    clip_pixels=hist.sum()*clip_limit
    clip_limit , clipped = calculate_clip_limit(hist,clip_pixels)
    excess = np.maximum(0, hist - clip_limit)
    e = np.sum(excess)
    hist_clipped =  np.minimum(hist, clip_limit)
            
    # Compute the probability density function (PDF) for values below the threshold
    
    C = hist[hist < clip_limit]
    if C.sum() ==0:
         return hist
    pk = hist / C.sum()
     # Create the adjusted histogram
    adjusted_histogram = np.copy(clipped)
    mask = adjusted_histogram < clip_limit
    a = clip_limit+1
    b= 0
    while a > clip_limit:
        adjusted_histogram =np.where(mask,hist_clipped+ pk * e,hist_clipped)
        prev_e = e
        e = np.sum(np.maximum(0, adjusted_histogram - clip_limit))
        hist_clipped =  np.minimum(adjusted_histogram, clip_limit)
        a = round(adjusted_histogram.max())
        if  b ==10:
            break
        elif e == prev_e and b <10:
            b+=1
            
    return adjusted_histogram

    
def apply_clahe_to_tile(tile , clip_limit):
    hist , bins = np.histogram(tile.flatten(), 65535, [0, 65535])
    new_hist = clip_histogram(hist , clip_limit)
    cdf = new_hist.cumsum()
    cdf_normalized =  np.floor((cdf - cdf.min()) / (cdf.max() - cdf.min()) *255).astype(np.uint8)
    
    #print(f'cdf max = {cdf_normalized.max()},cdf min={cdf_normalized.min()}')
    tile_equalized  = cdf_normalized[tile].astype("uint8")
    
    return tile_equalized , cdf_normalized
def De(temp):
    temp_hist, _ = np.histogram(temp.flatten(),65535,[0,65535])
    
    temp_hist = temp_hist / temp_hist.sum()

    # Step 4: Calculate entropy using the formula
    temp_hist = temp_hist[temp_hist > 0]
    entropy = -np.sum(temp_hist * np.log2(temp_hist))
    
    return entropy
    
def Psnr(temp,tile):
    mse = np.mean((tile - temp) ** 2)
    if mse == 0:
        return float('inf')  # Infinite PSNR for identical images
    return 20 * np.log10((65535) / math.sqrt(mse))

    
    
def compute_clr(tile,clr):
    clr_counter,optimumClr,maxDE,maxPSNR = 0,0,0,0
    for clr_counter in clr:
        temp,_ = apply_clahe_to_tile(tile, clr_counter)
        de =De(temp)
        psnr =Psnr(temp,tile)
        if de > maxDE and psnr >maxPSNR:
            optimumClr = clr_counter
            maxDE , maxPSNR = de , psnr
    #print(f'optCLR{optimumClr}')
    return optimumClr        
def woa(tile,clr):
    maxDE,maxPSNR ,optimumCl = 0,0,0
    a= clr/10 
    i=0
    while (i+1)* a < 99*a:
        temp_cl = (i+1)* a
        temp,_ = apply_clahe_to_tile(tile, temp_cl)
        de =De(temp)
        psnr =Psnr(temp,tile)
        if de > maxDE and psnr >maxPSNR:
            optimumCl= temp_cl
            maxDE , maxPSNR = de , psnr
            i+= 5
        i+=5
    #print(f'optimumCl, maxDE , maxPSNR: {optimumCl,maxDE , maxPSNR}')
    return optimumCl   

def clahe_apply(image , tile_grid_size ,clip_limit):
    h, w = image.shape
    tile_h, tile_w = h // tile_grid_size[0], w // tile_grid_size[1]
    clahe_image = np.zeros_like(image)
    tile_cdf_dict={}
    #process each tile
    listofCL = []
    for i in range(tile_grid_size[0]):
        for j in range(tile_grid_size[1]):
            x1 , y1 = i * tile_h , j * tile_w
            x2 , y2 = min(x1 + tile_h,h), min(y1 + tile_w,w)
            tile = image[x1:x2 , y1:y2]
            clr =[0.1000,0.0100,0.0010]
            opt_clr = compute_clr(tile,clr)
            clip_limit = woa(tile,opt_clr)
            clahe_tile, cdf_to_keep = apply_clahe_to_tile(tile, clip_limit)
            #update output
            clahe_image[x1:x2 , y1:y2]= clahe_tile
            tile_cdf_dict[(i, j)] = cdf_to_keep
            listofCL.append(clip_limit)
    #print(listofCL)
    output_image = np.zeros_like(image)
   
    for i in range(h) :
        for j in range(w) :
            
            tile_i = i // tile_h #1
            tile_j = j // tile_w #1
            
            tile_i *=  tile_h 
            tile_j *=  tile_w #top left 
            origin_value = image[i,j]
            
            #print(f'cdf max = {origin_cdf.max()},cdf min={origin_cdf.min()}')
            #print(F'org val = {origin_value} ,clahe value ={clahe_image[i,j]}, image={image[i,j]}')
            middle_i = tile_i + tile_h//2
            middle_w = tile_j + tile_w // 2
            #computing new tile
            a = 0.0
            
            if middle_i <= i and middle_w <= j:
                if  middle_i+tile_h < h and middle_w+tile_w < w :
                    a=2.0
                    min_x ,max_x ,min_y ,max_y= middle_i,middle_i+tile_h,middle_w,middle_w+tile_w
                    distA,distB,distC,distD =dist(i,j,min_x ,max_x ,min_y , max_y,a)
                elif  middle_i+tile_h >= h and middle_w+tile_w < w :
                    a =1.0
                    min_x ,max_x ,min_y ,max_y= middle_i,middle_i,middle_w,middle_w+tile_w
                    distA,distB =dist(i,j,min_x ,max_x ,min_y , max_y,a)
                elif  middle_i+tile_h < h and middle_w+tile_w >= w :
                    a=1.1
                    min_x ,max_x ,min_y ,max_y= middle_i,middle_i+tile_h,middle_w,middle_w
                    distA,distB =dist(i,j,min_x ,max_x ,min_y , max_y,a)
                elif  middle_i+tile_h >= h and middle_w+tile_w >= w :
                    output_image[i,j] = clahe_image[i,j]
                    continue
                #donedone
                    
            elif middle_i <= i and middle_w >= j:
                    if middle_i+tile_h < h and middle_w-tile_w >=0 :
                        a=2.0
                        min_x ,max_x ,min_y ,max_y=middle_i,middle_i+tile_h,middle_w-tile_w,middle_w
                        distA,distB,distC,distD =dist(i,j,min_x ,max_x ,min_y , max_y,a)
                    elif  middle_i+tile_h >= h and middle_w-tile_w >=0 :
                        a=1.0
                        min_x ,max_x ,min_y ,max_y=middle_i,middle_i,middle_w-tile_w,middle_w
                        distA,distB =dist(i,j,min_x ,max_x ,min_y , max_y,a)
                    elif  middle_i+tile_h < h and middle_w-tile_w <0 :
                        a=1.1
                        min_x ,max_x ,min_y ,max_y=middle_i,middle_i+tile_h,middle_w,middle_w
                        distA,distB =dist(i,j,min_x ,max_x ,min_y , max_y,a)
                    elif middle_i+tile_h >= h and middle_w-tile_w < 0 :
                        output_image[i,j] = clahe_image[i,j]
                        continue
                    #Done
            elif(middle_i >= i and middle_w <= j):
                if middle_i-tile_h>=0 and middle_w+tile_w<w-1 :
                    a=2.0
                    min_x ,max_x ,min_y , max_y =max(middle_i-tile_h,0),middle_i,middle_w,min(middle_w+tile_w,w-1)
                    distA,distB,distC,distD =dist(i,j,min_x ,max_x ,min_y , max_y,a)
                elif middle_i-tile_h<0 and middle_w+tile_w<w :
                    a = 1.0
                    min_x ,max_x ,min_y , max_y =middle_i,middle_i,middle_w,middle_w+tile_w
                    distA,distB =dist(i,j,min_x ,max_x ,min_y , max_y,a)
                elif middle_i-tile_h>=0 and middle_w+tile_w>=w :
                    min_x ,max_x ,min_y , max_y =middle_i-tile_h,middle_i,middle_w,middle_w
                    a =1.1
                    distA,distB =dist(i,j,min_x ,max_x ,min_y , max_y,a)
                elif middle_i-tile_h<0 and middle_w+tile_w>=w:
                    output_image[i,j] = clahe_image[i,j]
                    continue
                #done
            elif middle_i > i and middle_w > j:
                if middle_i-tile_h>=0 and middle_w-tile_w>=0:
                    a=2.0
                    min_x ,max_x ,min_y ,max_y=middle_i-tile_h,middle_i,middle_w-tile_w,middle_w
                    distA,distB,distC,distD =dist(i,j,min_x ,max_x ,min_y , max_y,a)                    
                elif middle_i-tile_h<0 and middle_w-tile_w>=0:
                    a  =1.0
                    min_x ,max_x ,min_y ,max_y=middle_i,middle_i,middle_w-tile_w,middle_w
                    distA,distB =dist(i,j,min_x ,max_x ,min_y , max_y,a)
                elif middle_i-tile_h>=0 and middle_w-tile_w<0:
                    a = 1.1
                    min_x ,max_x ,min_y ,max_y=middle_i-tile_h,middle_i,middle_w,middle_w
                    distA,distB =dist(i,j,min_x ,max_x ,min_y , max_y,a)
                elif middle_i-tile_h<0 and middle_w-tile_w<0:
                    output_image[i,j] = clahe_image[i,j]
                    continue
                #done
            #values and new pixel values
            if int(a)== 2:
                valueA, valueB, valueC, valueD = calc_value(origin_value,tile_h,tile_w,min_x ,max_x ,min_y , max_y,tile_cdf_dict,a)
                F , E = (valueA*distB[1] +valueB*distA[1])/tile_w ,  (valueC*distD[1] +valueD*distC[1])/tile_w
                new_pixel_value = (F*distC[0] + E*distA[0])/tile_h
                output_image[i,j]= new_pixel_value
                
            elif int(a) == 1:
                valueA, valueB = calc_value(origin_value,tile_h,tile_w,min_x ,max_x ,min_y , max_y,tile_cdf_dict,a)
                new_pixel_value = (valueA*distB + valueB * distA)/(distB +distA)
                output_image[i,j]= new_pixel_value
                
            else:
                continue
    #print(clahe_image.shape)
    return clahe_image,output_image
def calc_value(origin_value,tile_h,tile_w,min_x ,max_x ,min_y , max_y,tile_cdf_dict,a):
    if a ==2.0:
        #A
        tileA_h,tileA_w = min_x // tile_h,min_y // tile_w
        temp_cdf =tile_cdf_dict[(tileA_h,tileA_w)]
        valueA = temp_cdf[origin_value]
        #B
        tileB_h,tileB_w = min_x // tile_h,max_y // tile_w
        temp_cdf =tile_cdf_dict[(tileB_h,tileB_w)]
        valueB = temp_cdf[origin_value]
        #C
        tileC_h,tileC_w = max_x // tile_h,min_y // tile_w
        temp_cdf =tile_cdf_dict[(tileC_h,tileC_w)]
        valueC = temp_cdf[origin_value]
        #D
        tileD_h,tileD_w = max_x // tile_h,max_y // tile_w
        temp_cdf =tile_cdf_dict[(tileD_h,tileD_w)]
        valueD = temp_cdf[origin_value]
        return valueA,valueB,valueC,valueD
    elif int(a) ==1:
        tileA_h,tileA_w = min_x // tile_h,min_y // tile_w
        temp_cdf =tile_cdf_dict[(tileA_h,tileA_w)]
        valueA = temp_cdf[origin_value]
    #B
        tileB_h,tileB_w = max_x // tile_h,max_y // tile_w
        temp_cdf =tile_cdf_dict[(tileB_h,tileB_w)]
        valueB = temp_cdf[origin_value]
        return valueA,valueB
def dist(i,j,min_x ,max_x ,min_y , max_y,a):
    li = [i-min_x,i-max_x,j -min_y,j-max_y]
    if a == 2.0 and 0 not in li:
        distA =  abs(i-min_x) , abs(j -min_y)                  
        distB =  abs(i-min_x)  ,  abs(j-max_y)                
        distC =  abs(i-max_x),  abs(j -min_y)                   
        distD =   abs(i-max_x)  , abs(j-max_y)
        return distA,distB,distC,distD 
    elif  a == 2.0 and 0  in li:
        z = 0000.1
        distA =  abs(i-min_x+z) , abs(j -min_y+z)                  
        distB =  abs(i-min_x+z)  ,  abs(j-max_y+z)                
        distC =  abs(i-max_x+z),  abs(j -min_y+z)                   
        distD =   abs(i-max_x+z)  , abs(j-max_y+z)
        return distA,distB,distC,distD 
    elif a ==1.0:
        distA =  abs(j -min_y)                
        distB =  abs(j-max_y)   
        return distA,distB
    elif a ==1.1:          
        distA =  abs(i-min_x)                  
        distB =  abs(i-max_x) 
        return distA,distB  
    


image = cv2.imread('FLIR_00001.tiff', cv2.IMREAD_UNCHANGED)
clahe ,output_image = clahe_apply(image , (4,5), 5)
#print(f'after clahe:{output_image.max(),output_image.min(),output_image.sum(),output_image.size}')
blurred = cv2.GaussianBlur(image,(5,5),2)
output_image = (output_image * (image/(blurred))**15)
#print(f'after equation:{output_image.max(),output_image.min(),output_image.sum(),output_image.size}')
output_image = np.clip(output_image,0,255).astype(np.uint8)
#print(f'after clipping: {output_image.max(),output_image.min(),output_image.sum(),output_image.size}')
original_hist, _ =  np.histogram(image.flatten(), 65535, [0, 65535])
equalized_hist, _ = np.histogram(output_image.flatten(), 65535, [0, 65535])
de100 =De(output_image)
psnr100 =Psnr(output_image,image)

print(f'entropy and psnr:{de100, psnr100}')

#cv2.imwrite('D:/Finals/ACL_BBCE/image_2215/ACL_BBCE.png',output_image)
plt.subplot(1,2,1)
plt.imshow(blurred, cmap='gray')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(output_image, cmap='gray')
plt.axis('off')
plt.show()
