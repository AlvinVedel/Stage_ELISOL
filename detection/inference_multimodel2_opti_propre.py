import torch
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import math
import random
import torchvision
import time




mode = "test"

tag = "multires_v3_optimax"

save_box = True
folder_save = "pred_gros"

plot_predictions = False
compute_result = False
prefixe_save = "gros_"
total_tp = 0
total_fp = 0
total_fn = 0


if mode == "test" :
    images_folder = "../Test-10 photos_base/"
elif mode == "val" :
    images_folder = "/home/alvin/elisol/photos_entieres/val-entiere/"
elif mode == "all" :
    images_folder = ""


if mode == 'test' :
    images_names = [f for f in os.listdir(images_folder) if f.endswith(".jpg")]
elif mode == 'val' :
    images_names = [f for f in os.listdir(images_folder+"images/") if f.endswith(".jpg")]
elif mode == "all" :
    images_names = [("/home/alvin/elisol/photos_entieres/train-entiere/", f) for f in os.listdir("/home/alvin/elisol/photos_entieres/train-entiere/images/")] + [("/home/alvin/elisol/photos_entieres/val-entiere/", f) for f in os.listdir("/home/alvin/elisol/photos_entieres/val-entiere/images/")] +[("../Test-10 photos_base/", f) for f in os.listdir("../Test-10 photos_base/") if f.endswith(".jpg")]                    
    statu = [0 for f in os.listdir("/home/alvin/elisol/photos_entieres/train-entiere/images/")] + [1 for f in os.listdir("/home/alvin/elisol/photos_entieres/val-entiere/images/")] +[2 for f in os.listdir("../Test-10 photos_base/") if f.endswith(".jpg")]                    
mode = "correction"
if mode == "correction" :
    images_folder = "../correction/"
    images_names = [f for f in os.listdir(images_folder) if f.endswith(".jpg") and not os.path.exists(images_folder+f[:-3]+"txt")]

print(images_names)







### chargement du modèle




### hyper paramètres des résolutions, juste à modifier ici
de_hyp = True
if de_hyp : #conf, min, max, iou
    hyps = [0.65, 0, 650, 0.5,  ## params res 1   # environ 25-30 secondes pour tous les crops
            0.65, 600, 2000, 0.5,  ## params res 2  # environ 10 secondes pour tous les crops
            0.65, 1400, 10000, 0.5,  ## params res 3  # environ 3 secondes pour tous les crops
            0.48] # iou gobal
   
    resolutions = [2432, 1280, 640]
    models_paths = ["runs/train/rtx5_2432/weights/best.pt", "runs/train/rtx5_2432/weights/best.pt","runs/train/rtx5_2432/weights/best.pt"]
    overlap_forgetting = [(1, 1), (1, 1), (1, 1)] # 
    use_overlap_forgetting = True
    res_hyps = {}
    for i, res in enumerate(resolutions) :
        res_hyps[res] = {"conf":hyps[i*4], "min":hyps[i*4+1], "max":hyps[i*4+2], "nms":hyps[i*4+3]}

    res_hyps["global_nms"] = hyps[-1] 


temps_total = 0
nb_images = 0


# fonction pour extraire les crops en "parallèle" 
n_workers = 8
def process_row(i):
    my_index = i*len(x_starts)
    row = rows[i]
    y = y_starts[i]
    count_y = y_counts[i]
    
    for j in range(len(x_starts)):
        image_crops[my_index] = row[:, x_starts[j]:x_ends[j]]
        crop_pos[my_index] = [j, count_y, x_starts[j], y]
        my_index += 1
    return True

from concurrent.futures import ThreadPoolExecutor

# paramètres d'extraction de crops
split_width = 1820*3
split_height = 1214*4
overlap_x = 1820
overlap_y = 1214


for ii, img_name in enumerate(images_names) :
    t0 = time.time()
    print("STARTING", img_name, "...")
    if img_name == "15157.jpg" :
            print("celle la je la fais pas")
            continue
    if mode == "test" or mode=="correction":
            path = images_folder + img_name
    elif mode == "val" :
            path = images_folder + "images/" + img_name
    elif mode == "all" :
        path =img_name[0]+"images/" +img_name[1] if statu[ii] == 0 or statu[ii]==1 else img_name[0] +img_name[1]
    

    ## ouverture image
    image = cv2.imread(path)
    h_im, l_im = image.shape[:2]
    
    # extraction des crops
    x_starts = np.concatenate([np.arange(0, l_im-split_width, overlap_x, dtype=np.uint16), [l_im-split_width]], axis=0)
    x_ends = x_starts + split_width
    y_starts = np.concatenate([np.arange(0, h_im-split_height, overlap_y, dtype=np.uint16), [h_im -split_height]], axis=0)
    y_ends = y_starts+split_height
    rows = [image[y_starts[i]:y_ends[i]] for i in range(len(y_starts))]
    y_counts = np.arange(0, len(y_starts), 1, dtype=np.uint16)

        # tableau vide pour contenir les crops
    n_sample = len(x_starts)*len(y_starts)
    image_crops = np.empty((n_sample, split_height, split_width, 3), dtype=np.uint8)
    crop_pos = np.zeros((n_sample, 4))

        # extraction concurrentes des crops et écriture dans tableau vide
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        res = list(executor.map(process_row, range(len(rows))))

        # récupération indices max pour filtrage plus bas      
    max_x = np.max(crop_pos[:, 0])
    max_y = np.max(crop_pos[:, 1])
    print("temps de l'extraction :", time.time()-t0)  ### entre 3 et 5 secondes en général


    ### partie prédiction multi résolution
    global_images_boxes =None  # va contenir l'ensemble des boites
    for re, resolution in enumerate(resolutions) :
        # chargement du modèle (supprimable / 1 fois au départ si un seul modèle)
        model1 = torch.hub.load('./', 'custom', path=models_paths[re], source='local')
        model1.eval()
        model1.max_det = 1000
        # récupération des hyper paramètres associés à chaque résolution
        borne_inf = res_hyps[resolution]["min"]
        borne_sup = res_hyps[resolution]["max"]
        confv = res_hyps[resolution]["conf"]
        nmsv = res_hyps[resolution]["nms"]
        model1.conf = confv
        model1.iou = nmsv

        resolution_prediction = None # contenir ensemble des prédictins de la résolution (tous crops)
        for r, crop_r in enumerate(image_crops) :
            idx, idy, x, y = crop_pos[r]
            if use_overlap_forgetting : # système d'oubli de certains crops qui permet de ne pas prédire avec overlap maximal : réduit énormément le temps mais baisse un peu le rappel (F1 monte car plus de précision par contre)
                # default (1, 1) = overlap maximum ou on regarde tous les crops
                forget_y, forget_x = overlap_forgetting[re]
                if not ((idx%forget_x == 0 and idx!=(max_x-1)) and ((idy%forget_y)==0 and idy!=(max_y-1))) :
                    #print("j'oublie le crop", idx, idy)
                    continue
                

            # prédiction du crop à redimensionnement (resolution, resolution), le no_grad important pour la mémoire / temps
            with torch.no_grad():
                results = model1(crop_r, augment=False, size=(resolution, resolution))

            # conserve toutes les boxes sur gpu pour y faire les opération
            boxes = results.xywh[0]  

            # Filtrage 1 : taille + confiance
            w, h = boxes[:, 2], boxes[:, 3]
            sum_wh = w + h
            conf = boxes[:, 4]

            mask1 = (sum_wh >= borne_inf) & (sum_wh < borne_sup) & (conf >= confv)
            boxes = boxes[mask1]

            # Passage xywh en xyxy 
            x_center, y_center = boxes[:, 0], boxes[:, 1]
            w, h = boxes[:, 2], boxes[:, 3]
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2
            conf = boxes[:, 4]

            boxes_xyxy = torch.stack([x1, y1, x2, y2, conf], dim=-1)

            # Filtrage 2 : pas bord du crop sauf si premier ou dernier crop
            infos_crops = torch.tensor(crop_pos[r], device=boxes.device).repeat(len(boxes_xyxy), 1)
            split_width_t = torch.tensor(split_width, device=boxes.device)
            split_height_t = torch.tensor(split_height, device=boxes.device)
            max_x_t = torch.tensor(max_x, device=boxes.device)
            max_y_t = torch.tensor(max_y, device=boxes.device)

            valids2 = (
                ((boxes_xyxy[:, 0] / split_width_t > 0.05) | (infos_crops[:, 0] == 0)) &
                ((boxes_xyxy[:, 2] / split_width_t < 0.95) | (infos_crops[:, 0] == max_x_t)) &
                ((boxes_xyxy[:, 1] / split_height_t > 0.05) | (infos_crops[:, 1] == 0)) &
                ((boxes_xyxy[:, 3] / split_height_t < 0.95) | (infos_crops[:, 1] == max_y_t))
            )

            boxes_xyxy = boxes_xyxy[valids2]


            # Filtrage 3 : nms si boxes ayant passé filtre
            if len(boxes_xyxy) > 0:
                keep = torchvision.ops.nms(boxes_xyxy[:, :4], boxes_xyxy[:, 4], nmsv)
                boxes_xyxy = boxes_xyxy[keep]

            # callage x, y sur image globale
            boxes_xyxy[:, [0, 2]] += x
            boxes_xyxy[:, [1, 3]] += y

            # ajout dans les prédiciton de l'image pour résolution re
            if len(boxes_xyxy) > 0 :
                if resolution_prediction is None :
                    resolution_prediction = boxes_xyxy
                else :
                    resolution_prediction = torch.concat([resolution_prediction, boxes_xyxy], 0)

        # filtrage nms pour toute la résolution et ajout dans prédiction globale
        if resolution_prediction is not None :
            res = torchvision.ops.nms(resolution_prediction[:, :4], resolution_prediction[:, 4], nmsv)
            resolution_prediction = resolution_prediction[res]
            colonne_res = torch.ones(size=(resolution_prediction.shape[0], 1), device=resolution_prediction.device) * resolution
            resolution_prediction = torch.concat([resolution_prediction, colonne_res], 1)
            if global_images_boxes is None :
                global_images_boxes = resolution_prediction
            else :
                global_images_boxes = torch.concat([global_images_boxes, resolution_prediction], 0)
            #global_images_boxes += nms_boxes([b for b in resolution_prediction if b[4] > confv], nmsv)
            print("Fin resolution", re, "temps depuis début :",time.time()-t0, "nb box prédites :", len(resolution_prediction))
        else :
            print("Fin resolution", re, "temps depuis début :",time.time()-t0, "nb box prédites :", 0)

    # tout dernier filtrage : nms globale cross-resolution
    if global_images_boxes is None :
      print("aucune boite prédite")
      continue  
    keep = torchvision.ops.nms(global_images_boxes[:, :4], global_images_boxes[:, 4], res_hyps["global_nms"])

    global_images_boxes = global_images_boxes[keep].cpu().numpy() # retour sur cpu
    print("temps de prédiction total :", time.time()-t0)
    temps_total += time.time()-t0
    nb_images += 1
    print("temps moyen :", tag, temps_total/nb_images)






    if save_box :
        save_path = "/home/alvin/elisol/multires_output/"+mode+"/"+folder_save+"/"+img_name[:-4]+".csv"

        # en pixels
        xcs = (global_images_boxes[:, 0] + global_images_boxes[:, 2])/2  /l_im
        ycs = (global_images_boxes[:, 1] + global_images_boxes[:, 3])/2   / h_im
        ws = (global_images_boxes[:, 2] - global_images_boxes[:, 0])   / l_im
        hs = (global_images_boxes[:, 3] - global_images_boxes[:, 1])  / h_im

        #imhs = (np.ones((len(xcs))) * h_im).astype(np.uint32)
        #imls = (np.ones(len(xcs)) *l_im).astype(np.uint32)
        cls_id = np.zeros((len(xcs)), dtype=np.uint8)

        with open(images_folder+img_name[:-3]+"txt", "w") as f:
            for i in range(len(xcs)) :
                f.write(f"{cls_id[i]} {xcs[i]:.6f} {ycs[i]:.6f} {ws[i]:.6f} {hs[i]:.6f} \n")
        


        #df_results = pd.DataFrame({"cls":cls_id, "xc":xcs, "yc":ycs, "w":ws, "h":hs, "conf":global_images_boxes[:, 4], "res":global_images_boxes[:, 5].astype(np.uint32), "hauteur_im":imhs, "largeur_im":imls})
        #df_results.to_csv(save_path, index=False)


    if plot_predictions or compute_result:
        # calcul métriques et plot
        def iou(box1, box2):
            x1, y1, x2, y2 = box1
            x1b, y1b, x2b, y2b = box2
            inter_x1 = max(x1, x1b)
            inter_y1 = max(y1, y1b)
            inter_x2 = min(x2, x2b)
            inter_y2 = min(y2, y2b)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            area_box1 = (x2 - x1) * (y2 - y1)
            area_box2 = (x2b - x1b) * (y2b - y1b)
            union_area = area_box1 + area_box2 - inter_area
            return inter_area / union_area if union_area > 0 else 0



        gt_boxes = []
        if mode == 'test' :
            path_ = "inference/"+img_name[:-4]+'.txt'
        elif mode == "val" :
            path_ = images_folder+"labels_v2/"+img_name[:-4]+'.txt'
        with open(path_, 'r') as f:
            for line in f:
                parts = line.strip().split()
                _, xc, yc, w, h = map(float, parts)
                x1 = int((xc - w / 2) * l_im)
                y1 = int((yc - h / 2) * h_im)
                x2 = int((xc + w / 2) * l_im)
                y2 = int((yc + h / 2) * h_im)

                gt_boxes.append([x1, y1, x2, y2])


        used_preds = np.zeros((len(global_images_boxes)))
      
        matched_boxes = []
        false_neg = []
        false_pos = []

        
        #print("compte tp")
        if len(global_images_boxes) > 0 :
            for box in gt_boxes :
                ious = [iou(box, [p[0], p[1], p[2], p[3]])  if used_preds[i]==0 else 0 for i, p in enumerate(global_images_boxes)]   # calcule la liste des iou avec toutes les boxes de prédictions  ==> si pred déjà utilisée alors 0
                best_box_ind = np.argmax(ious)
                if ious[best_box_ind] > 0.5 :
                            ### l'IOU est supérieur à 0.8 avec vérité terrain => BONNE DETECTION
                    used_preds[best_box_ind] = 1
                    matched_boxes.append(global_images_boxes[best_box_ind])
                    
                else :
                    ### aucune ne match la GT donc nématode non reconnu ==> FAUX NEGATIF  
                    false_neg.append(box)
                   
            for i in range(len(global_images_boxes)) :
                if used_preds[i] == 0 :
                    ## on a prédit un truc qui n'a pas match avec la GT  ==> FAUX POSITIF
                    false_pos.append(global_images_boxes[i])
                   
        else :
            for box in gt_boxes :
                false_neg.append(box)   ### si on ne retient aucune box prédite, que des Faux négatifs

        tp = len(matched_boxes)
        fp = len(false_pos)
        fn = len(false_neg)

        p = tp / max(tp+fp, 1)
        r = tp / max(tp+fn, 1)
        f1 = 2*p*r / max(p+r, 1e-6)
        print("resultats", img_name, " : P =", p, "R =", r, "F1 =", f1)
        total_tp+=tp
        total_fp += fp
        total_fn += fn

        p = total_tp / max(total_tp+total_fp, 1)
        r = total_tp / max(total_tp+total_fn, 1)
        f1 = 2*p*r / max(p+r, 1e-6)
        print("resultats globaux : P =", p, "R =", r, "F1 =", f1)

        if plot_predictions :
            for i, box in enumerate(matched_boxes): 
                x1, y1, x2, y2, conf, re = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 25*re, 0), 2)  # BLEU
                cv2.putText(image, str(round(float(conf), 2))+"- w="+str(int(x2-x1))+" - h="+str(int(y2-y1))+" Res="+str(re), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            ### PLOT DES FAUX NEGATIF
            for i, box in enumerate(false_neg): 
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # VERT
                cv2.putText(image, "GT - w="+str(int(x2-x1))+" - h="+str(int(y2-y1)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
            ### PLOT DES FAUX POSITIFS
            for i, box in enumerate(false_pos): 
                x1, y1, x2, y2, conf, re = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 25*re, 255), 2)  # ROUGE
                cv2.putText(image, str(round(float(conf), 2))+"- w="+str(int(x2-x1))+" - h="+str(int(y2-y1))+" Res="+str(re), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imwrite("inference/"+prefixe_save + img_name[:-4] + "_"+tag+".jpg", image)


    
        