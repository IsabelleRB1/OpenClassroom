# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 18:20:45 2021

@author: cyril
"""

import streamlit as st
import matplotlib.pyplot as plt

import numpy as np

import sys




def _scale_data(data, ranges):
    (x1, x2) = ranges[0]
    d = data[0]
    res = [(d-y1) / (y2-y1) * (x2-x1) + x1
           for d, (y1, y2) in zip(data, ranges)]
    return res


class RadarChart():
    def __init__(self, fig, variables, ranges,
                     n_ordinate_levels=6):
            angles = np.arange(0, 360, (360. / len(variables)))

            axes = [fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True,
                                 label="axes{}".format(i))
                    for i in range(len(variables))]

            axes[0].set_thetagrids(angles, labels=[])

            for ax in axes[1:]:
                ax.patch.set_visible(False)
                ax.grid("off")
                ax.xaxis.set_visible(False)

            for i, ax in enumerate(axes):
                grid = np.linspace(*ranges[i],
                                   num=n_ordinate_levels)
                gridlabel = ["{}".format(round(x, 2))
                             for x in grid]
                if ranges[i][0] > ranges[i][1]:
                    grid = grid[::-1]  # hack to invert grid
                    # gridlabels aren't reversed
                gridlabel[0] = ""  # clean up origin
                ax.set_rgrids(grid, labels=gridlabel, angle=angles[i], fontsize=6)
                # ax.spines["polar"].set_visible(False)
                ax.set_ylim(*ranges[i])

            ticks = angles
            ax.set_xticks(np.deg2rad(ticks))  # crée les axes suivant les angles, en radians
            ticklabels = variables
            ax.set_xticklabels(ticklabels, fontsize=1)  # définit les labels

            angles1 = np.linspace(0, 2 * np.pi, len(ax.get_xticklabels()) + 1)
            angles1[np.cos(angles1) < 0] = angles1[np.cos(angles1) < 0] + np.pi
            angles1 = np.rad2deg(angles1)
            labels = []
            for label, angle in zip(ax.get_xticklabels(), angles1):
                x, y = label.get_position()
                lab = ax.text(x, y - .5, label.get_text(), transform=label.get_transform(),
                              ha=label.get_ha(), va=label.get_va())
                lab.set_rotation(angle)
                lab.set_fontsize(9)
                lab.set_fontweight('bold')
                labels.append(lab)
            ax.set_xticklabels([])

            # variables for plotting
            self.angle = np.deg2rad(np.r_[angles, angles[0]])
            self.ranges = ranges
            self.ax = axes[0]


    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        l = self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
        return l

    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def legend(self, categories, *args, **kw):
        self.ax.legend(*args, **kw)

    def title(self, title, *args, **kw):
        self.ax.text(0.9, 1, title, transform=self.ax.transAxes, *args, **kw)
        
def radar_plot(cust_features, desc_features_train, categories, xsize=0.1, ysize=0.05):
    features_train_1 = []
    features_train_0 = []
    
    # List of descriptive variables
    var_descriptives = [col for col in categories]
    
    
    features_train_1 = desc_features_train[desc_features_train['Difficulties payment']=='yes'][categories].mean()
    features_train_0 = desc_features_train[desc_features_train['Difficulties payment']=='no'][categories].mean()
    cust_features_sel = cust_features[categories] 
    
    # Min,max descriptive varaibles
    var_descript_ranges = [list(desc_features_train[categories].describe().loc[['min', 'max'], var])
                           for var in var_descriptives]
   
 

   
    fig = plt.figure(figsize = (6, 6))
    lax = []
    data_plot = np.array(cust_features_sel)
    radar = RadarChart(fig, var_descriptives,
                           var_descript_ranges)
    l, = radar.plot(data_plot, color='b', linewidth=1.0)
    lax.append(l)
    radar.fill(data_plot, alpha=0.2, color='b')
    data_plot = np.array(features_train_0)
    l, = radar.plot(data_plot, color='g', linewidth=1.0)
    lax.append(l)
    data_plot = np.array(features_train_1)
    l, = radar.plot(data_plot, color='r', linewidth=1.0)
    lax.append(l)
    
    # Titre du radarchart

    #radar.title(title='test',
    #           color='r',
    #           size=14)
  
    radar.ax.legend(handles = lax, labels=['Customer','Credit granted','Credit not granted'], loc=3, bbox_to_anchor=(0,0,1,1), bbox_transform=fig.transFigure,prop={'size': 6} )
    # radar.legend()
    st.pyplot(fig)
