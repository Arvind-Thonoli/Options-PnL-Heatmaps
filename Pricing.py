import math
from jax import P
from matplotlib import pyplot as plt
import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from colorspacious import cspace_converter

st.title("Options Heatmap")
x = 10 #number of rows and coloumns for the heatmap 
st.sidebar.markdown("[Arvind Thonoli]('www.linkedin.com/in/arvind-bijulal/')")
#INPUTS
with st.sidebar:
    Strike_Price = st.number_input('Strike Price',min_value=1, value = 50,step=1)
    TTM = st.number_input('Time to Maturity in years',min_value=1/12, value = 1.0,step=1/12)
    RF = st.number_input('Risk Free rate',min_value=0.01,max_value=1.0, value = 0.2,step=0.01)
    Vmax = st.number_input("Maximum volatility",min_value=0.01,value=1.0)
    SPmax = st.number_input("Maximum stock price", min_value =1.00,value = 100.00)
    V2 = st.slider("Volatility Range", 0.01, Vmax, (0.01, 1.0))
    SP2 = st.slider("Stock Price Range", 0.01, SPmax, (25.0, 75.0))
    Call_PP = st.number_input('Call Price for PnL',min_value=0.01, value = 10.0,step=0.01)
    Put_PP = st.number_input('Put Price for PnL',min_value=0.01, value = 5.0,step=0.01)
st.sidebar.markdown("Input Option Prices to calculate Profitability at different Stock Prices and Volatilites")
Volatility_array = np.array([])
for i in range (x+1):
    Volatility_array = np.append(Volatility_array,np.percentile(V2,i*(100/x)))

SP2_array = np.array([])
for i in range (x+1):
    SP2_array = np.append(SP2_array,np.percentile(SP2,i*(100/x)))

#Call Price Calculation
call_price_array = np.empty([x+1,x+1])
for i in range(len(Volatility_array)):
    for j in range(len(SP2_array)):
        d1 = float((math.log(SP2_array[j] / Strike_Price) + (RF+(Volatility_array[i]**2/2))*TTM)/(Volatility_array[i]*TTM**0.5))
        d2 = float(d1 - (Volatility_array[i] * TTM**2))
        call_price_array[j,i] = ((SP2_array[j] * norm.cdf(d1)) - ((Strike_Price * math.exp(-RF * TTM)) * norm.cdf(d2)))

#PNL Call Calculation
PNL_call_array = np.empty([x+1,x+1])
for i in range(len(Volatility_array)):
    for j in range(len(SP2_array)):
        PNL_call_array[j,i] = call_price_array[j,i] - Call_PP
print(PNL_call_array)

#Put Price Calculation
put_price_array = np.empty([x+1,x+1])
for i in range(len(Volatility_array)):
    for j in range(len(SP2_array)):
        d1 = float((math.log(SP2_array[j] / Strike_Price) + (RF+(Volatility_array[i]**2/2))*TTM)/(Volatility_array[i]*TTM**0.5))
        d2 = float(d1 - (Volatility_array[i] * TTM**2))
        put_price_array[j,i] = ((Strike_Price * math.exp(-RF * TTM)) * norm.cdf(-d2)) - (SP2_array[j] * norm.cdf(-d1))
        
#PNL Put Calulation
PNL_put_array = np.empty([x+1,x+1])
for i in range(len(Volatility_array)):
    for j in range(len(SP2_array)):
        PNL_put_array[j,i] = put_price_array[j,i] - Put_PP

#Heatmap customizations Call Price
st.set_page_config(layout = "wide")
fig, (ax,ax2) = plt.subplots(figsize = (18,7), ncols =2)

ax.set_yticks(range(len(SP2_array)), labels=np.round(SP2_array,2),
              rotation=45, ha="right", rotation_mode="anchor")
ax.set_xticks(range(len(Volatility_array)), labels=np.round(Volatility_array,2))

for i in range(len(Volatility_array)):
    for j in range(len(SP2_array)):
        text = ax.text(i, j, round(call_price_array[j, i], 2),
                       ha="center", va="center", color="black")

ax.set_title("Call Price")
img = ax.imshow(call_price_array, cmap= plt.cm.cool)
fig.tight_layout()
ax.set_ylabel('Stock Price')
ax.set_xlabel('Volatility')
fig.colorbar(img, ax=ax, orientation="vertical")
ax.invert_yaxis()

#Heatmap customizations Put Price

ax2.set_yticks(range(len(SP2_array)), labels=np.round(SP2_array,2),
              rotation=45, ha="right", rotation_mode="anchor")
ax2.set_xticks(range(len(Volatility_array)), labels=np.round(Volatility_array,2))

for i in range(len(Volatility_array)):
    for j in range(len(SP2_array)):
        text = ax2.text(i, j,round(put_price_array[j, i],2),
                       ha="center", va="center", color="black")

ax2.set_title("Put Price")
img2 = ax2.imshow(put_price_array, cmap=plt.cm.cool)
fig.colorbar(img2, ax=ax2, orientation="vertical")
fig.tight_layout()
ax2.invert_yaxis()
ax2.set_xlabel('Volatility')
ax2.set_ylabel('Stock Price')
st.pyplot(plt.gcf())

st.header("PnL Heatmap")
#Heatmap customizations PNL Call Price
st.set_page_config(layout = "wide")
fig, (ax3,ax4) = plt.subplots(figsize = (18,7), ncols =2)

ax3.set_yticks(range(len(SP2_array)), labels=np.round(SP2_array,2),
              rotation=45, ha="right", rotation_mode="anchor")
ax3.set_xticks(range(len(Volatility_array)), labels=np.round(Volatility_array,2))

for i in range(len(Volatility_array)):
    for j in range(len(SP2_array)):
        text = ax3.text(i, j,round(PNL_call_array[j, i],2),
                       ha="center", va="center", color="black")

ax3.set_title("PnL Call Price")

# Helper function to choose norm and cmap
def get_pnl_norm_cmap(data):
    vmin, vmax = np.min(data), np.max(data)
    if vmin < 0 < vmax:
        # Data spans zero: use diverging colormap and TwoSlopeNorm
        cmap = LinearSegmentedColormap.from_list("pnl_cmap", ["#ff0000", "#ffff00", "#00ff00"])
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    elif vmax <= 0:
        # All values <= 0: use red to yellow
        cmap = LinearSegmentedColormap.from_list("neg_cmap", ["#ff0000", "#ffff00"])
        norm = Normalize(vmin=vmin, vmax=0)
    else:
        # All values >= 0: use yellow to green
        cmap = LinearSegmentedColormap.from_list("pos_cmap", ["#ffff00", "#00ff00"])
        norm = Normalize(vmin=0, vmax=vmax)
    return norm, cmap

# For PNL Call
norm_call, cmap_call = get_pnl_norm_cmap(PNL_call_array)
img3 = ax3.imshow(PNL_call_array, cmap=cmap_call, norm=norm_call)
fig.colorbar(img3, ax=ax3, orientation="vertical")
fig.tight_layout()
ax3.set_ylabel('Stock Price')
ax3.set_xlabel('Volatility')
ax3.invert_yaxis()

#Heatmap customizations PNL Put Price

ax4.set_yticks(range(len(SP2_array)), labels=np.round(SP2_array,2),
              rotation=45, ha="right", rotation_mode="anchor")
ax4.set_xticks(range(len(Volatility_array)), labels=np.round(Volatility_array,2))

for i in range(len(Volatility_array)):
    for j in range(len(SP2_array)):
        text = ax4.text(i, j,round(PNL_put_array[j, i],2),
                       ha="center", va="center", color="black")

ax4.set_title("PnL Put Price")
norm_put, cmap_put = get_pnl_norm_cmap(PNL_put_array)
img4 = ax4.imshow(PNL_put_array, cmap=cmap_put, norm=norm_put)
fig.colorbar(img4, ax=ax4, orientation="vertical")
fig.tight_layout()
ax4.invert_yaxis()
ax4.set_xlabel('Volatility')
ax4.set_ylabel('Stock Price')
st.pyplot(plt.gcf())