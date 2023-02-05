import copy
import csv
import functools
import glob
import os
from collections import namedtuple
import SimpleITK as sitk
import numpy as np
import torch
import torch.cuda
from torch.utils.data import Dataset
from util.disk import getCache
from util.util import XyzTuple, xyz2irc
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('part2ch10_raw')
CandidateInfoTuple = namedtuple( #用來記錄某個位置是否為結節
    'CandidateInfoTuple',  #樣本tuple名稱
    'isNodule_bool, diameter_mm,series_uid,center_xyz', #該tuple所包含的資訊(用來記錄某個位置是否為結節、直徑、序號、座標)

)
@functools.lru_cache(1) #記憶體內快取
def getCandidateInfoList(requireOnDisk_bool=True):#偵測那些UID已被放入硬碟
    mhd_list=glob.glob('Luna_Data/subset*/*.mhd')
    presentOnDisk_set={os.path.split(p)[-1][:-4] for p in mhd_list}
    #把annotations按UID分組，之後要與candidates合併
    diameter_dict={}#建立字典存真結結的座標與直徑
    with open('luna/annotations.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:#跳過第一列
            series_uid=row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])#xyz座標整理成tuple
            annotationDiameter_mm=float(row[4])#第四行資料，代表直徑大小
            diameter_dict.setdefault(series_uid,[]).append((annotationCenter_xyz,annotationDiameter_mm))#存到字典中，key為UID，value為(座標，直徑)
    #建立完整的候選結點樣本串列
    candidateInfo_list = []
    with open('luna/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid=row[0]
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:#若找不到UID，表示未被下載至硬碟，應跳過
                continue
            isNodule_bool = bool(int(row[4]))#取得是否為結點
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])#xyz座標

            candidateDiameter_mm=0.0
            for annotation_tup in diameter_dict.get(series_uid, []): #處理真結結的樣本
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                for i in range(3): #依次走訪xyz
                    delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])#兩CSV中有相同UID的座標中心點的差距
                    if delta_mm > annotationDiameter_mm / 4:#檢查是否差異過大，過大則視為不同結結
                        break
                else:
                    candidateDiameter_mm = annotationDiameter_mm#更新候選結節的大小
                    break
            candidateInfo_list.append(CandidateInfoTuple( #資料存入list中
                isNodule_bool,
                candidateDiameter_mm,
                series_uid,
                candidateCenter_xyz,
            ))    
    candidateInfo_list.sort(reverse=True)#真結結會被排到前面，非結結隨後
    return candidateInfo_list


#把CT轉成python
class Ct:
    def __init__(self,series_uid):
        mhd_path = glob.glob('Luna_Data/subset*/{}.mhd'.format(series_uid))[0]
        ct_mhd=sitk.ReadImage(mhd_path)#用sitk來匯入資料
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        #處理離群值
        ct_a.clip(-1000,1000,ct_a)#把資料限制在只有-1000到1000之間

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())#取得原點篇一輛
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())#取得體素大小
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)#方向轉為陣列

#把資料切成小塊來篩選
#傳入中心座標和體素大小，回傳CT切塊和轉化的中心位置
    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(#先改成IRC
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        slice_list=[]#進行切塊
        for axis, center_val in enumerate(center_irc):#依序走訪IRC
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])
            #例外處理
            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])
            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc

@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc

class LunaDataset(Dataset):
    def __init__(self,
                val_stride=10,#樣本存入驗證集的頻率
                isValSet_bool=None,
                series_uid=None,
    ):
        self.candidateInfo_list=copy.copy(getCandidateInfoList())#複製以免影響原來資料
        if series_uid:
            self.candidateInfo_list=[
                x for x in self.candidateInfo_list if x.series_uid==series_uid
            ]
        if isValSet_bool:#True時，建立驗證集
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]#每隔10筆就抽一筆來當驗證資料
            assert self.candidateInfo_list
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]#每隔10筆就刪一筆
            assert self.candidateInfo_list

    def __len__(self):
        return len(self.candidateInfo_list)#回傳樣本數

    def __getitem__(self, ndx):
        candidateInfo_tup=self.candidateInfo_list[ndx]
        width_irc=(32,48,48)
        candidate_a,center_irc=getCandidateInfoList( #傳回值得shap(32,48,48)分別為切片數量、列述、行數
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc,
        )
        candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)#加入通道數

        #建立用來表示是否為結節的張量
        pos_t=torch.tensor(
            [
                not candidateInfo_tup.isNodule_bool, #若是節節則為[0,1]，反之[1,0]
                candidateInfo_tup.isNodule_bool
            ],
            dtype=torch.long,
        )

        return(
            candidate_t,
            pos_t,
            candidateInfo_tup.series_uid,
            torch.tensor(center_irc)
        )
