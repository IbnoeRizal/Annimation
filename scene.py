from __future__ import annotations
from manim import *
import numpy as np

x_min,x_max = -5,5
y_min,y_max = -5,5

np.random.seed(1194)
data = np.random.randint(low=x_min,high=x_max+1,size=(20,2))



class Kmeans:
    def __init__(self,data:np.ndarray,kcentroid:int = 2) -> None:
        self.datacoord = data
        self.k = max(2,kcentroid)

    def _get_euclidean(self):

        if self.centroid is None or self.datacoord is None:
            return self
        
        self.alldistance = np.linalg.norm(self.centroid[None,:,:] - self.datacoord[:,None,:],axis=-1)
        return self
        
    
    def _create_centroid(self):
        dims = list(self.datacoord.shape)
        dims[0] = self.k

        minim = np.amin(self.datacoord,axis=0)
        maxim = np.amax(self.datacoord,axis=0)

        self.centroid = np.around(np.random.uniform(low=minim,high=maxim,size=tuple(dims)),decimals=3)
        return self

    
    def _grouping(self):
        if self.alldistance is None or self.centroid is None:
            return self
        
        group = self.alldistance.argmin(axis=-1).astype(np.int32)
        self.k_group = group

        return self
            

    def _move_centroid(self):
        if self.centroid is None or self.alldistance is None or self.k_group is None:
            return None
        candidate = np.zeros_like(self.centroid)

        for k in range(self.k):
            mask = self.k_group == k
            candidate[k] = (self.datacoord[mask]).mean(axis=0) if mask.any() else self.centroid[k]

        return candidate
    
    def fit(self,epsilon=1e-8,patience=3,iterate=30):
        original = patience
        if self.centroid is None:
            self._create_centroid()
        
        for i in range(iterate):
            candidate = self._get_euclidean()._grouping()._move_centroid()
            if candidate is None:
                return None
            
            deltaD = np.absolute(self.centroid - candidate).sum(axis=1).max(axis=0)
            self.centroid = candidate

            if deltaD <= epsilon:
                patience -= 1
            else:
                patience = original

            if patience <= 0:
                return None
            
            yield self

        


class KmeansScene(Scene):
    def construct(self) -> None:
        ax = Axes(
            x_range=[x_min,x_max,1],
            y_range=[y_min,y_max,1],
            x_length=12,
            y_length=7,
            axis_config={
                "include_numbers": True,
            },
            tips=True
        )

        datapoints = VGroup([Dot(point=ax.coords_to_point(coord),radius=DEFAULT_DOT_RADIUS,color=RED) for coord in data])

        self.add(ax)
        self.play(LaggedStart(*[FadeIn(dot) for dot in datapoints],lag_ratio=0.1))

        kmean = Kmeans(data=data,kcentroid=3)
        kmean._create_centroid()

        #centroid
        centroid = VGroup([Dot(ax.c2p(coord),radius=DEFAULT_DOT_RADIUS*1.5,color=WHITE) for coord in kmean.centroid])

        self.play(LaggedStart(*[FadeIn(dot) for dot in centroid],lag_ratio=0.2))

        kmean._get_euclidean()._grouping()
        lines = []
        for i in range(3):
            anggota = [dot for dot, lab in zip(datapoints, kmean.k_group) if lab == i]
            lines.append(VGroup([Line(start=p,end=centroid[i]) for p in anggota]))
            self.play(Create(lines[i],lag_ratio=0.4))

        self.wait(0.5)

        self.remove(*lines)
        del lines

        for km in kmean.fit(patience=4,iterate=20):
            for i in range(len(km.centroid)):
                self.play(centroid[i].animate.move_to(ax.c2p(km.centroid[i])))

            lines = []
            for i in range(3):
                anggota = [dot for dot, lab in zip(datapoints, kmean.k_group) if lab == i]
                lines.append(VGroup([Line(start=p,end=centroid[i]) for p in anggota]))
                self.play(Create(lines[i],lag_ratio=0.4))

            self.wait(0.5)
            self.remove(*lines)