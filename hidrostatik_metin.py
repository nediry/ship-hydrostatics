"""
Created on Fri May 21 22:38:13 2021
@author: nedir ymamov
"""

import numpy as np
from scipy.integrate import cumtrapz

boy = 113 # GEMİNİN BOYU (LBP)
genislik = 17.38 # GEMİNİN TAM GENİŞLİĞİ
draft = 6.68 # GEMİNİN TAM YÜKLÜ DRAFTI
offset = np.loadtxt("s60_cb70.txt", dtype = float) # BOYUTSUZ OFFSET TABLOSU
satir, sutun = offset.shape # OFFSET TABLOSUNU BOYUTU

# posta: TÜM POSTALARIN KIÇTAN KONUMLARI, suhatti: TÜM SUHATTININ DİPTEN KONUMLARI
posta = np.array([0, .5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.5, 10]) * boy / 10
suhatti0 = np.array([0, .3, 1, 2, 3, 4, 4 * 1.325, 4 * 1.65]) * draft / 4 # PRU
suhatti = np.array([0, .5, 1, 2, 3, 4, 4 * 1.325, 4 * 1.65]) * draft / 4 # PRU
# suhatti0 = np.array([0, .3, 1, 2, 3, 4, 5, 6]) * draft / 4  # YTU
# suhatti = np.array([0, .5, 1, 2, 3, 4, 5, 6]) * draft / 4  # YTU
# suhatti0 = np.array([0, .3, 1, 2, 3, 4, 4 * 1.35, 4 * 1.7]) * draft / 4 # ITU
# suhatti = np.array([0, .5, 1, 2, 3, 4, 4 * 1.35, 4 * 1.7]) * draft / 4 # ITU

# VERİLEN OFFSETTE 0.5 SUHATTI OLMADIĞINDAN İNTERPOLASYON YAPILDI
for i in range(satir):
    offset[i, :] = np.interp(suhatti, suhatti0, offset[i, :]) # YENİ BOYUTSUZ OFFSET TABLOSU

offset *= genislik / 2 # BOYUTLU OFFSET TABLOSUNU ELDE ETMEK İÇİN BOYUTSUZ OFFSET TABLOSUNU YARI GENİŞLİKLE ÇARPILDI
alan = np.zeros((satir, sutun))   # BON-JEAN ALANLARI
for i in range(satir):
    alan[i, 1:] = 2 * cumtrapz(offset[i, :], suhatti)

# suhatti2 = np.array([0, .5, 1, 2, 3, 4, 4 * 1.35, 4 * 1.7]) * 30 #ITU
suhatti2 = np.array([0, .5, 1, 2, 3, 4, 4 * 1.325, 4 * 1.65]) * 32.9416453 #PRU
# suhatti2 = np.array([0, .5, 1, 2, 3, 4, 5, 6]) * 32.9416453  #YTU
alan_oran = np.max(alan) / 37.3
alan_yeni = alan / alan_oran
print("_Curve")
for i in range(8):
    print(str(alan_yeni[1, i] + .5 * 37.4074) + ',' + str(suhatti2[i]))
print("enter")
print("_Curve")
for i in range(8):
    print(str(alan_yeni[1, i] + 9.5 * 37.4074) + ',' + str(suhatti2[i]))
print("enter")
alan_yeni = np.delete(alan_yeni, [1, 11], 0)
dx = 0
for i in range(10):
    print("_Curve")
    for j in range(8):
        print(str(alan_yeni[i, j] + dx) + ',' + str(suhatti2[j]))
    print("enter")
    dx += 37.4074


moment = np.zeros((satir, sutun))  # BON-JEAN MOMENTLERİ
for i in range(sutun):
    moment[:, i] = offset[:, i] * suhatti[i]
for i in range(satir):
    moment[i, 1:] = 2 * cumtrapz(moment[i, :], suhatti)

moment_oran = np.max(moment) / 37.2 + 1
moment_yeni = moment / moment_oran
print("_Curve")
for i in range(8):
    print(str(moment_yeni[1, i] + .5 * 37.4074) + ',' + str(suhatti2[i]))
print("enter")
print("_Curve")
for i in range(8):
    print(str(moment_yeni[1, i] + 9.5 * 37.4074) + ',' + str(suhatti2[i]))
print("enter")
moment_yeni = np.delete(moment_yeni, [1, 11], 0)
dx = 0
for i in range(10):
    print("_Curve")
    for j in range(8):
        print(str(moment_yeni[i, j] + dx) + ',' + str(suhatti2[j]))
    print("enter")
    dx += 37.4074


hacim = np.zeros(sutun)  # HACİM HESABI
for i in range(1, sutun):
    hacim[i] = np.trapz(alan[:, i], posta)

hacim_oran = hacim[-1] / 374.07 + 2
hacim_yeni = hacim / hacim_oran
print("_Curve")
for i in range(sutun):
    print(str(hacim_yeni[i]) + ',' + str(suhatti2[i]))
print("enter")


deplasman = 1.025 * hacim  # DEPLASMAN HESABI

deplasman_oran = deplasman[-1] / 374.07
deplasman_yeni = deplasman / deplasman_oran
print("_Curve")
for i in range(sutun):
    print(str(deplasman_yeni[i]) + ',' + str(suhatti2[i]))
print("enter")


AWP = np.zeros(sutun)  # SU HATTI ALANI
for i in range(sutun):
    AWP[i] = 2 * np.trapz(offset[:, i], posta)

AWP_oran = AWP[-1] / 374.07
AWP_yeni = AWP / AWP_oran
print("_Curve")
for i in range(sutun):
    print(str(AWP_yeni[i]) + ',' + str(suhatti2[i]))
print("enter")


LCF = np.zeros(sutun)  # YÜZME MERKEZİNİN BOYUNA YERİ (kıçtan)
for i in range(sutun):  # LCF = MxAWP / AWP
    LCF[i] = 2 * np.trapz(offset[:, i] * posta, posta) / AWP[i]

LCF_oran = boy / 374.074
LCF_yeni = LCF / LCF_oran
print("_Curve")
for i in range(sutun):
    print(str(LCF_yeni[i]) + ',' + str(suhatti2[i]))
print("enter")


LCB = np.zeros(sutun)  # HACİM MERKEZİNİZ BOYUNA YERİ (kıçtan)
for i in range(1, sutun):  # LCB = Mxalan / hacim
    LCB[i] = np.trapz(alan[:, i] * posta, posta) / hacim[i]

#oran = boy / 374.074
LCB_yeni = LCB / LCF_oran
print("_Curve")
for i in range(1, sutun):
    print(str(LCB_yeni[i]) + ',' + str(suhatti2[i]))
print("enter")


T1 =  AWP * 1.025 / 100  # 1cm BATMA TONAJI

T1_oran = 1.5 * 37.4074 / np.max(T1)
T1_yeni = T1 * T1_oran
print("_Curve")
for i in range(sutun):
    print(str(T1_yeni[i]) + ',' + str(suhatti2[i]))
print("enter")


CM = np.zeros(sutun)  # ORTA KESİT NARİNLİK KATSAYISI
CM[1:] = alan[6, 1:] / (2 * offset[6, 1:] * suhatti[1:])

oran = 37.407 * .5 / (np.max(CM) + .1)
CM_yeni = CM * oran
print("_Curve")
for i in range(1, sutun):
    print(str(CM_yeni[i]) + ',' + str(suhatti2[i]))
print("enter")


CB = np.zeros(sutun)
CB[1:] = hacim[1:] / (boy * 2 * offset[6, 1:] * suhatti[1:]) # BLOK KATSAYISI

CB_oran = 37.4074 * .5 / (np.max(CB) + 1)
CB_yeni = CB * CB_oran
print("_Curve")
for i in range(1, sutun):
    print(str(CB_yeni[i]) + ',' + str(suhatti2[i]))
print("enter")

CW = AWP / (boy * 2 * offset[6, :])  # SU HATTI NARİNLİK KATSAYISI

oran = 37.4074 * .5 / (np.max(CW) + .1)
CW_yeni = CW * oran
print("_Curve")
for i in range(1, sutun):
    print(str(CW_yeni[i] + 37.4074) + ',' + str(suhatti2[i]))
print("enter")


CP = np.zeros(sutun)
CP[1:] = CB[1:] / CM[1:]  # PRİZMATİK KATSAYI

oran = 37.407 * .5 / (np.max(CP) + .1)
CP_yeni = CP * oran
print("_Curve")
for i in range(1, sutun):
    print(str(CP_yeni[i] + 37.4074 * .5) + ',' + str(suhatti2[i]))
print("enter")


KB = np.zeros(sutun) # HACİM MERKEZİNİN DÜŞEY YERİ
for i in range(1, sutun):
    KB[i] = np.trapz(moment[:, i], posta) / hacim[i]

KB_oran = 1.5 * 37.4074 / np.max(KB)
KB_yeni = KB * KB_oran
print("_Curve")
for i in range(1, sutun):
    print(str(KB_yeni[i]) + ',' + str(suhatti2[i]))
print("enter")


ICL = np.zeros(sutun)  # ORTA SİMETRİ EKSENİNE GÖRE ATALET MOMENTİ
for i in range(sutun):
    ICL[i] = (2 / 3) * np.trapz(offset[:, i]**3, posta)
BM = np.zeros(sutun)  # ENİNE METESANTR YARIÇAPI
BM[1:] = ICL[1:] / hacim[1:]

BM_yeni = BM * KB_oran
print("_Curve")
for i in range(1, sutun):
    print(str(KB_yeni[i] + BM_yeni[i]) + ',' + str(suhatti2[i]))
print("enter")


Im = np.zeros(sutun)  # MASTORİYE GÖRE ATALET MOMENTİ
moment_kolu = np.array([-5, -4.5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 4.5, 5]) * boy / 10
for i in range(8):
    Im[i] = 2 * np.trapz(offset[:, i] * moment_kolu**2, posta)

# SU HATTI ALANININ YÜZME MERKEZİNDEN GEÇEN EKSENE GÖRE ATALET MOMENTİ
If = Im - AWP * (LCF - boy / 2)**2
BML = np.zeros(sutun)  # BOYUNA METASANTR YARIÇAPI
BML[1:] = If[1:] / hacim[1:]

BML_oran = np.max(np.abs(BML)) / 374.074
BML_yeni = BML / BML_oran
print("_Curve")
for i in range(1, sutun):
    print(str(BML_yeni[i]) + ',' + str(suhatti2[i]))
print("enter")


MCT1 = np.zeros(sutun)
MCT1[1:] = deplasman[1:] * BML[1:] / (100 * boy)  # BİR SANTİM TRİM MOMENTİ

MTC1_oran = 3.5 * 37.4074 / np.max(MCT1)
MCT1_yeni = MCT1 / MTC1_oran
print("_Curve")
for i in range(1, sutun):
    print(str(MCT1_yeni[i]) + ',' + str(suhatti2[i]))
print("enter")


# ISLAK YÜZEY ALAN EĞRİSİ