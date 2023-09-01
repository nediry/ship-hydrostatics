"""
Created on Fri May 21 20:44:37 2021
@author: nedir ymamov
"""

import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz
import dataframe_image as dfi

boy = 123 # GEMİNİN BOYU (LBP)
genislik = 17.571 # GEMİNİN TAM GENİŞLİĞİ
draft = 11.949 # GEMİNİN TAM YÜKLÜ DRAFTI
yogunluk = 1.025 # DENİZ SUYUNUN YOĞUNLUĞU
offset = np.loadtxt("s60_cb70.txt", dtype = float) # BOYUTSUZ OFFSET TABLOSU
satir, sutun = offset.shape # OFFSET TABLOSUNUN BOYUTU

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

moment = np.zeros((satir, sutun))  # BON-JEAN MOMENTLERİ
for i in range(sutun):
    moment[:, i] = offset[:, i] * suhatti[i]
for i in range(satir):
    moment[i, 1:] = 2 * cumtrapz(moment[i, :], suhatti)

hacim = np.zeros(sutun)  # HACİM HESABI
for i in range(1, sutun):
    hacim[i] = np.trapz(alan[:, i], posta)

deplasman = yogunluk * hacim  # DEPLASMAN HESABI

AWP = np.zeros(sutun)  # SU HATTI ALANI
for i in range(sutun):
    AWP[i] = 2 * np.trapz(offset[:, i], posta)

LCF = np.zeros(sutun)  # YÜZME MERKEZİNİN BOYUNA YERİ (KIÇTAN)
for i in range(sutun):  # LCF = MxAwp / Awp
    LCF[i] = np.trapz(offset[:, i] * posta, posta) / AWP[i] - boy / 2

LCB = np.zeros(sutun)  # HACİM MERKEZİNİZ BOYUNA YERİ (KIÇTAN)
for i in range(1, sutun):  # LCB = Mxalan / hacim
    LCB[i] = np.trapz(alan[:, i] * posta, posta) / hacim[i] - boy / 2

T1 =  AWP * yogunluk / 100  # 1cm BATMA TONAJI

CM = np.zeros(sutun)  # ORTA KESİT NARİNLİK KATSAYISI
CM[1:] = alan[6, 1:] / (2 * offset[6, 1:] * suhatti[1:])

CB = np.zeros(sutun)
CB[1:] = hacim[1:] / (boy * 2 * offset[6, 1:] * suhatti[1:]) # BLOK KATSAYISI

CW = AWP / (boy * 2 * offset[6, :])  # SU HATTI NARİNLİK KATSAYISI

CP = np.zeros(sutun)
CP[1:] = CB[1:] / CM[1:]  # PRİZMATİK KATSAYI

KB = np.zeros(sutun) # HACİM MERKEZİNİN DÜŞEY YERİ
for i in range(1, sutun):
    KB[i] = np.trapz(moment[:, i], posta) / hacim[i]

ICL = np.zeros(sutun)  # ORTA SİMETRİ EKSENİNE GÖRE ATALET MOMENTİ
for i in range(sutun):
    ICL[i] = (2 / 3) * np.trapz(offset[:, i]**3, posta)
BM = np.zeros(sutun)  # ENİNE METESANTR YARIÇAPI
BM[1:] = ICL[1:] / hacim[1:]

IM = np.zeros(sutun)  # MASTORİYE GÖRE ATALET MOMENTİ
for i in range(sutun):
    IM[i] = np.trapz(offset[:, i] * posta**2, posta)

# SU HATTI ALANININ YÜZME MERKEZİNDEN GEÇEN EKSENE GÖRE ATALET MOMENTİ
IF = IM - AWP * (boy / 2 - LCF)**2

BMl = np.zeros(sutun)  # BOYUNA METASANTR YARIÇAPI
BMl[1:] = IF[1:] / hacim[1:]

MCT1 = np.zeros(sutun)
MCT1[1:] = deplasman[1:] * BMl[1:] / (100 * boy)  # BİR SANTİM TRİM MOMENTİ

# ISLAK YÜZEY ALAN EĞRİSİ
def arc_length(x, y):
    npts = len(x)
    arc = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
    for k in range(1, npts):
        arc += np.sqrt((x[k] - x[k - 1])**2 + (y[k] - y[k - 1])**2)
    return arc / 2

l = np.zeros((satir, sutun))
for i in range(satir):
    for j in range(1, sutun):
        l[i, j] = round(arc_length(offset[i, j - 1 : j + 1], suhatti[j - 1 : j + 1]), 3)
S = np.zeros(sutun)
for i in range(1, sutun):
    S[i] = S[i - 1] + 2 * np.trapz(l[:, i], posta)

df = pd.DataFrame([hacim, deplasman, LCB, LCF, CB, CM, CP, CW, AWP, T1, MCT1, ICL, IM, IF, KB, BM, BMl, S],
                  columns=["WL0", "WL0.5", "WL1", "WL2", "WL3", "WL4", "WL5", "WL6"],
                  index=["V", "dep", "LCB", "LCF", "CB", "CM", "CP", "CWP", "AWP", "T1", "MT1", "Icl", "Im", "If", "KB", "BM", "BMl", "S"])

df = df.round(2)
dfi.export(df, "hidrostatik_tablo.png")