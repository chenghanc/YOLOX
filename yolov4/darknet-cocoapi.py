from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import matplotlib.pyplot as plt
import itertools
from tabulate import tabulate

COCO_CLASSES = (
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T", "U", "W", "X", "Y", "Z",
    "c0", "c1", "c2", "c3","c4","c5","c6","c7","c8","c9","cA","cB","cC","cD","cE","cF","cG","cH","cI","cJ","cK","cL","cM","cN","cO","cP",
    "cQ","cR","cS","cT","cU","cV","cW","cX","cY","cZ","cCN","cMO","cHK","h0", "h1","h2","h3","h4","h5","h6","h7","h8","h9","hA","hB",
    "hC","hD","hE","hF","hG","hH","hJ","hK","hL","hM","hN","hP","hR", "hS","hT","hU","hV","hW","hX","hY","hZ",
)

def per_class_AP_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AP"], colums=6):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1] # AP@[ IoU=0.50:0.95 ]
        #precision = precisions[0, :, idx, 0, -1] # AP@[ IoU=0.50 ]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table

def per_class_AR_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AR"], colums=6):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table

annType = 'bbox'

anno_file = 'darknet_valid_gt_baby.json'
results_json = 'darknet_pred_baby.json'

cocoGt=COCO(anno_file)
cocoDt=cocoGt.loadRes(results_json)

imgIds=sorted(cocoDt.getImgIds())
len(imgIds)

# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
#cocoEval.params.catIds = [5]                # COCO API evaluation for subset of classes
cocoEval.params.imgIds = imgIds
#cocoEval.params.maxDets = [1, 100, 1000] # reset maxDets for Baby
#cocoEval.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2,155 ** 2], [155** 2,185 ** 2], [185** 2, 1e5 ** 2]] # reset object area for AffectNet
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

cat_ids = list(cocoGt.cats.keys())
cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]

AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
AR_table = per_class_AR_table(cocoEval, class_names=cat_names)

infoAP = "per class AP:\n" + AP_table + "\n"
infoAR = "per class AR:\n" + AR_table + "\n"

print(infoAP)
print(infoAR)

#print(cocoEval.stats)

# pr curve
#  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
#  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
#  catIds     - [all] K cat ids to use for evaluation
#  areaRng    - [...] A=4 object area ranges for evaluation
#  maxDets    - [1 10 100] M=3 thresholds on max detections per image

#  counts     - [T,R,K,A,M] parameter dimensions
#  precision  - [TxRxKxAxM] precision for every evaluation setting
#  recall     - [TxKxAxM] max recall for every evaluation setting

'''
cocoEval.eval['counts']
cocoEval.eval['params']
cocoEval.eval['date']
cocoEval.params.iouThrs

all_precision = cocoEval.eval['precision'][0, :, :, 0, 2] # data for IoU@0.50
#all_precision = cocoEval.eval['precision'][5, :, :, 0, 2] # data for IoU@0.75
len(all_precision)
#all_precision
all_recall = cocoEval.params.recThrs
len(all_recall)
#all_recall
cocoEval.eval['scores'][0, :, 0, 0, 2]

names = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T", "U", "W", "X", "Y", "Z",
    "c0", "c1", "c2", "c3","c4","c5","c6","c7","c8","c9","cA","cB","cC","cD","cE","cF","cG","cH","cI","cJ","cK","cL","cM","cN","cO","cP",
    "cQ","cR","cS","cT","cU","cV","cW","cX","cY","cZ","cCN","cMO","cHK","h0", "h1","h2","h3","h4","h5","h6","h7","h8","h9","hA","hB",
    "hC","hD","hE","hF","hG","hH","hJ","hK","hL","hM","hN","hP","hR", "hS","hT","hU","hV","hW","hX","hY","hZ",
]

x = np.arange(0, 1.01, 0.01)
if 0 < len(names) < 108:
    print(" \n" + " AP@[ IoU=0.50 ]" + " (%)")
    print(" ***********************")
    for i, y in enumerate(all_precision.T):
        print(' Category :  {0}  : {1:.2f}  -  {2}'.format(i,all_precision[:,i].mean() * 100,names[i]))
        plt.plot(x, y, linewidth=1, label=f'{names[i]} {all_precision[:,i].mean():.3f}')  # plot(recall, precision)
else:
    plt.plot(x, all_precision, linewidth=1, color='grey')                                 # plot(recall, precision)

print(" -----")
print(" All Categories : %.2f" % (all_precision.mean() * 100) + "\n" + " ***********************" + "\n")

print(" \n" + " AP@[ IoU=0.50:0.95 ]" + " (%)")
print(" ***********************")
ap = cocoEval.eval['precision']
num_classes = 107
avg_ap = 0.0
for i in range(0, num_classes):
    # 0:all 1:small 2:medium 3:large
    s = ap[:,:,i,0,2]
    print(' Category :  {0}  : {1:.2f}  -  {2}'.format(i,s.mean() * 100,names[i]))
    avg_ap += s.mean() * 100
print(" -----")
print(" All Categories : {:.2f}".format(avg_ap / num_classes) + "\n" + " ***********************" + "\n")

plt.plot(x, all_precision.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.50' % all_precision.mean())
plt.title('PR Curve: mAP@0.50 =  %.3f' % all_precision.mean())
plt.ylabel("Precision")
plt.xlabel("Recall")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.savefig('prcurve.jpg',dpi=250)
plt.show()
'''