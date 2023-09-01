"""
Created on Fri May 28 23:23:04 2021
@author: nedir ymamov
"""

import numpy as np
import pandas as pd
import dataframe_image as dfi
from scipy.integrate import cumtrapz


class GemiHidrostatigi:    
    def __init__(self, boy, genislik, draft, offset, posta, suhatti, yogunluk):
        self.boy = boy
        self.genislik = genislik
        self.draft = draft
        self.offset = offset * self.genislik / 2
        self.posta = posta * self.boy
        self.suhatti = suhatti * self.draft
        self.yogunluk = yogunluk
    
    
    def offset_goster(self):
        sutun = np.linspace(0, len(self.suhatti) - 1, len(self.suhatti))
        satir = np.linspace(0, len(self.posta) - 1, len(self.posta))
        df = pd.DataFrame(self.offset, columns = sutun, index = satir)
        return df
    

    def offset_genisletme(self, satir, sutun):
        # suhatti_yeni: TÜM SUHATTININ DİPTEN KONUMLARI
        suhatti_yeni = np.linspace(0, 6, sutun) * self.draft / 4
        # posta_yeni: TÜM POSTALARIN KIÇTAN KONUMLARI
        posta_yeni = np.linspace(0, 10, satir) * self.boy / 10
        
        offset2 = np.zeros((self.offset.shape[0], sutun))
        for i in range(self.offset.shape[0]):
            offset2[i, :] = np.interp(suhatti_yeni, self.suhatti, self.offset[i, :])
        self.offset = np.zeros((satir, sutun))
        for i in range(self.offset.shape[1]):
            self.offset[:, i] = np.interp(posta_yeni, self.posta, offset2[:, i])
        self.offset = np.round(self.offset, 3)
        self.posta = posta_yeni
        self.suhatti = suhatti_yeni
    

    def alan_hesabi(self):
        alan = np.zeros((self.offset.shape))   # BON-JEAN ALANLARI
        for i in range(self.offset.shape[0]):
            alan[i, 1:] = 2 * cumtrapz(self.offset[i, :], self.suhatti)
        return np.round(alan, 3)
    

    def moment_hesabi(self):
        moment = np.zeros((self.offset.shape))  # BON-JEAN MOMENTLERİ
        for i in range(self.offset.shape[1]):
            moment[:, i] = offset[:, i] * suhatti[i]
        for i in range(self.offset.shape[0]):
            moment[i, 1:] = 2 * cumtrapz(moment[i,:], suhatti)
        return np.round(moment, 3)
    

    def volume_hesabi(self):
        alan = self.alan_hesabi()
        hacim = np.zeros(self.offset.shape[1])  # HACİM HESABI
        for i in range(1, self.offset.shape[1]):
            hacim[i] = np.trapz(alan[:, i], self.posta)
        return np.round(hacim, 3)
    

    def deplasman_hesabi(self):
        hacim = self.volume_hesabi()
        deplasman = self.yogunluk * hacim  # DEPLASMAN HESABI
        return deplasman
    

    def AWP_hesabi(self):
        AWP = np.zeros(self.offset.shape[1])  # SU HATTI ALANI
        for i in range(self.offset.shape[1]):
            AWP[i] = 2 * np.trapz(self.offset[:, i], self.posta)
        return np.round(AWP, 3)
    

    def LCF_hesabi(self):
        AWP = self.AWP_hesabi()
        # YÜZME MERKEZİNİN BOYUNA YERİ (kıçtan)
        LCF = np.zeros(self.offset.shape[1])  
        for i in range(self.offset.shape[1]):  # LCF = MxAwp / Awp
            LCF[i] = np.trapz(self.offset[:, i] * self.posta, self.posta) / AWP[i] - self.boy / 2
        return np.round(LCF, 3)
    

    def LCB_hesabi(self):
        alan = self.alan_hesabi()
        hacim = self.volume_hesabi()
        # HACİM MERKEZİNİZ BOYUNA YERİ (kıçtan)
        LCB = np.zeros(self.offset.shape[1])
        for i in range(1, self.offset.shape[1]):  # LCB = Mxalan / hacim
            LCB[i] = np.trapz(alan[:, i] * self.posta, self.posta) / hacim[i] - self.boy / 2
        return np.round(LCB, 3)
    

    def T1_hesabi(self):
        AWP = self.AWP_hesabi()
        T1 =  AWP * yogunluk / 100  # 1cm BATMA TONAJI
        return T1
    

    def katsayi_hesabi(self):
        # ORTA KESİT NARİNLİK KATSAYISI
        alan = self.alan_hesabi()
        CM = np.zeros(self.offset.shape[1])
        CM[1:] = alan[6, 1:] / (2 * self.offset[6, 1:] * self.suhatti[1:])
        
        # BLOK KATSAYISI
        hacim = self.volume_hesabi()
        CB = np.zeros(self.offset.shape[1])
        CB[1:] = hacim[1:] / (self.boy* 2 * self.offset[6, 1:] * self.suhatti[1:])
        
        # SU HATTI NARİNLİK KATSAYISI
        AWP = self.AWP_hesabi()
        CW = AWP / (self.boy * 2 * self.offset[6, :])
        
        # PRİZMATİK KATSAYI
        CP = np.zeros(self.offset.shape[1])
        CP[1:] = CB[1:] / CM[1:]
        return CM, CB, CW, CP
    

    def KB_hesabi(self):
        moment = self.moment_hesabi()
        hacim = self.volume_hesabi()
        KB = np.zeros(self.offset.shape[1]) # HACİM MERKEZİNİN DÜŞEY YERİ
        for i in range(1, self.offset.shape[1]):
            KB[i] = np.trapz(moment[:, i], self.posta) / hacim[i]
        return KB
    

    def BM_hesabi(self):
        hacim = self.volume_hesabi()
        # ORTA SİMETRİ EKSENİNE GÖRE ATALET MOMENTİ
        ICL = np.zeros(self.offset.shape[1])
        for i in range(self.offset.shape[1]):
            ICL[i] = (2 / 3) * np.trapz(self.offset[:, i]**3, self.posta)
        BM = np.zeros(self.offset.shape[1])  # ENİNE METESANTR YARIÇAPI
        BM[1:] = ICL[1:] / hacim[1:]
        return ICL, BM
    

    def BML_hesabi(self):
        Im = np.zeros(self.offset.shape[1])  # MASTORİYE GÖRE ATALET MOMENTİ
        for i in range(self.offset.shape[1]):
            Im[i] = np.trapz(self.offset[:, i] * self.posta**2, self.posta)
        # SU HATTI ALANININ YÜZME MERKEZİNDEN GEÇEN EKSENE GÖRE ATALET MOMENTİ
        LCF = self.LCF_hesabi()
        AWP = self.AWP_hesabi()
        If = Im - AWP * (self.boy / 2 - LCF)**2
        BML = np.zeros(self.offset.shape[1])  # BOYUNA METASANTR YARIÇAPI
        hacim = self.volume_hesabi()
        BML[1:] = If[1:] / hacim[1:]
        return Im, If, BML
    

    def MTC1_hesabi(self):
        deplasman = self.deplasman_hesabi()
        _, _, BML = self.BML_hesabi()
        # BİR SANTİM TRİM MOMENTİ
        MCT1 = np.zeros(self.offset.shape[1])
        MCT1[1:] = deplasman[1:] * BML[1:] / (100 * self.boy)
        return MCT1
    

    def S_hesabi(self):
        # ISLAK YÜZEY ALAN EĞRİSİ
        def egri_boy(x, y):
            npts = len(x)
            egri = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
            for k in range(1, npts):
                egri += np.sqrt((x[k] - x[k - 1])**2 + (y[k] - y[k - 1])**2)
            return egri / 2

        l = np.zeros((self.offset.shape))
        for i in range(self.offset.shape[0]):
            for j in range(1, self.offset.shape[1]):
                l[i, j] = round(egri_boy(self.offset[i, j - 1 : j + 1], self.suhatti[j - 1 : j + 1]), 3)
        S = np.zeros(self.offset.shape[1])
        for i in range(1, self.offset.shape[1]):
            S[i] = S[i - 1] + 2 * np.trapz(l[:, i], self.posta)
        return S
    

    def tabloyu_kaydet(self):
        hacim = self.volume_hesabi()
        deplasman = self.deplasman_hesabi()
        LCB = self.LCB_hesabi()
        LCF = self.LCF_hesabi()
        CM, CB, CW, CP = self.katsayi_hesabi()
        AWP = self.AWP_hesabi()
        T1 = self.T1_hesabi()
        MCT1 = self.MTC1_hesabi()
        ICL, BM = self.BM_hesabi()
        IM, IF, BML = self.BML_hesabi()
        KB = self.KB_hesabi()
        S = self.S_hesabi()
        df = pd.DataFrame([hacim, deplasman, LCB, LCF, CB, CM, CP, CW, AWP, T1, MCT1, ICL, IM, IF, KB, BM, BML, S],
                          columns=["WL0", "WL0.5", "WL1", "WL2", "WL3", "WL4", "WL5", "WL6"],
                          index=["V", "dep", "LCB", "LCF", "CB", "CM", "CP", "CWP", "AWP", "T1", "MT1", "ICL", "Im", "If", "KB", "BM", "BML", "S"])
        
        df = df.round(2)
        dfi.export(df, "hidrostatik_tablo_2.png")
        return df

boy = 123 # GEMİNİN BOYU (LBP)
genislik = 17.571 # GEMİNİN TAM GENİŞLİĞİ
draft = 11.949 # GEMİNİN TAM YÜKLÜ DRAFTI
# posta: TÜM POSTALARIN KIÇTAN KONUMLARIN VERECEK KATSAYI VEYA MOMENT KOLU
posta = np.array([0, .5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.5, 10]) / 10
# suhatti: TÜM SUHATTININ DİPTEN KONUMLARI VERECEK KATSAYI VEYA MOMENT KOLU
suhatti = np.array([0, .3, 1, 2, 3, 4, 5, 6]) / 4
yogunluk = 1.025 # DENİZ SUYUNUN YOĞUNLUĞU
offset = np.loadtxt("s60_cb70.txt", dtype = float) # BOYUTSUZ OFFSET TABLOSU

gemi1 = GemiHidrostatigi(boy, genislik, draft, offset, posta, suhatti, yogunluk)
gemi1.tabloyu_kaydet()