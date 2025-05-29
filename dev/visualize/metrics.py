import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def arrowed_spines(ax=None, arrowLength=10, labels=(None, None), arrowStyle='<|-'):
    xlabel, ylabel = labels

    for i, spine in enumerate(['left', 'bottom']):
        # Set up the annotation parameters
        t = ax.spines[spine].get_transform()
        xy, xycoords = [0.99, 0], ('axes fraction', t)
        xytext, textcoords = [arrowLength, 0], ('offset points', t)
        
        # create arrowprops
        arrowprops = dict( arrowstyle=arrowStyle,
                           facecolor=ax.spines[spine].get_facecolor(), 
                        #    linewidth=ax.spines[spine].get_linewidth(),
                        #    alpha = ax.spines[spine].get_alpha(),
                           alpha = 1.,
                        #    zorder=ax.spines[spine].get_zorder(),
                           linestyle = ax.spines[spine].get_linestyle() )
    
        if spine == 'bottom':
            ha, va = 'left', 'center'
            xarrow = ax.annotate(xlabel, xy, xycoords=xycoords, xytext=xytext, 
                        textcoords=textcoords, ha=ha, va='center',
                        arrowprops=arrowprops)
        else:
            ha, va = 'center', 'bottom'
            yarrow = ax.annotate(ylabel, xy[::-1], xycoords=xycoords[::-1], 
                        xytext=xytext[::-1], textcoords=textcoords[::-1], 
                        ha='center', va=va, arrowprops=arrowprops)
    return xarrow, yarrow

# ------------------------- HV -------------------------- #
fig, ax = plt.subplots(1, 1, figsize=(5, 6.5))
plt.subplots_adjust(left=0.1, right=0.95, top=0.74, bottom=0.07)
ax.set_xticks([]) 
ax.set_yticks([]) 

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['left'].set_position(('axes', 0))

ax.set_xbound(0, 1)
ax.set_ybound(0, 1)
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)

ax.set_xlabel("Objective 1", fontsize=18)
ax.set_ylabel("Objective 2", fontsize=18)

arrowed_spines(ax)

x = np.linspace(0, 1, 1000)
def superellipse(x, n):
    return (1 - x**n)**(1/n)
y = np.array([superellipse(xi, 3) for xi in x])

front_points_x = np.array([0.2, 0.5, superellipse(0.8, 3), superellipse(0.5, 3)])
front_points_y = np.array([superellipse(xi, 3) for xi in front_points_x])

ax.vlines(front_points_x, 0, front_points_y, linestyle='dashed', color='dimgrey')
ax.hlines(front_points_y, 0, front_points_x, linestyle='dashed', color='dimgrey')

for i, (xi, yi) in enumerate(zip(front_points_x, front_points_y)):
    ax.fill_between([0, xi], 0, yi, color='lightgrey', alpha=1, label="Hyper Volume" if i == 0 else None)

ax.plot(x, y, color='black', label="Pareto Front")

ax.scatter(front_points_x, front_points_y, color='royalblue', s=150, linewidths=2, facecolors='royalblue',
           edgecolors='white', zorder=10, label="Pareto Approximation")


ax.scatter(0, 0, color='royalblue', s=200, marker='^', clip_on=False, zorder=10, edgecolors="white", label="Reference Point")
# fig.legend(loc=1)
# ax.scatter(front_points_x, front_points_y, color='royalblue', s=100)

fig.legend(loc="upper right", prop={'size': 15})
plt.savefig("plots/HV_new.pdf")

# ---------------------------- SP ------------------------------- #
fig, ax = plt.subplots(1, 1, figsize=(5, 6.5))
plt.subplots_adjust(left=0.1, right=0.95, top=0.74, bottom=0.07)
ax.set_xticks([]) 
ax.set_yticks([]) 

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['left'].set_position(('axes', 0))

ax.set_xbound(0, 1)
ax.set_ybound(0, 1)
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)

ax.set_xlabel("Objective 1", fontsize=18)
ax.set_ylabel("Objective 2", fontsize=18)

arrowed_spines(ax)

x = np.linspace(0, 1, 1000)
def superellipse(x, n):
    return (1 - x**n)**(1/n)
y = np.array([superellipse(xi, 3) for xi in x])

front_points_x = np.array([0.2, 0.5, superellipse(0.8, 3), superellipse(0.5, 3)])
front_points_y = np.array([superellipse(xi, 3) for xi in front_points_x])

ax.plot(x, y, color='black', label="Pareto Front")

ax.scatter(front_points_x, front_points_y, color='royalblue', s=150, linewidths=2, facecolors='royalblue',
           edgecolors='white', zorder=10, label="Pareto Approximation")

# for i, (xi, yi) in enumerate(zip(front_points_x, front_points_y)):
#     ax.fill_between([0, xi], 0, yi, color='lightgrey', alpha=1, label="Hyper Volume" if i == 0 else None)

# ax.plot(x, y, color='black', label="Pareto Front")

# ax.scatter(front_points_x, front_points_y, color='royalblue', s=150, linewidths=2, facecolors='royalblue',
#            edgecolors='white', zorder=10, label="Pareto Approximation")

for i in range(3):
    annotate = ax.annotate("", xy=(front_points_x[i], front_points_y[i]), xytext=(front_points_x[i+1], front_points_y[i+1]),
            arrowprops=dict(arrowstyle="<|-|>",
                            edgecolor='crimson',
                            facecolor='crimson',
                            lw=2,
                            shrinkA=2,
                            shrinkB=2),zorder=11, label="Sparsity")
    
# ax.scatter(0, 0, color='royalblue', s=200, marker='^', clip_on=False, zorder=10, edgecolors="white", label="Reference Point")
# fig.legend(loc=1)
# ax.scatter(front_points_x, front_points_y, color='royalblue', s=100)
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.patches import FancyArrowPatch
class AnnotationHandler(HandlerLine2D):
    def __init__(self,ms,*args,**kwargs):
        self.ms = ms
        HandlerLine2D.__init__(self,*args,**kwargs)
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)
        ydata = ((height - ydescent) / 2.) * np.ones(len(xdata), float)
        legline = FancyArrowPatch(posA=(xdata[0],ydata[0]),
                                  posB=(xdata[-1],ydata[-1]),
                                  arrowstyle=orig_handle.arrowprops["arrowstyle"],
                                  mutation_scale=self.ms,
                                  edgecolor=orig_handle.arrowprops["edgecolor"],
                                  facecolor=orig_handle.arrowprops["facecolor"],
                                  lw=1,
                                  shrinkA=2,
                                  shrinkB=2,)
        legline.set_transform(trans)
        return legline,

h, l = ax.get_legend_handles_labels()
fig.legend(handles = h +[annotate], 
          handler_map={type(annotate) : AnnotationHandler(5)},
          loc=1,
          prop={'size': 15})

plt.savefig("plots/SP_new.pdf")

# ---------------------- CO ---------------------- #
fig, ax = plt.subplots(1, 1, figsize=(5, 6.5))
plt.subplots_adjust(left=0.1, right=0.95, top=0.74, bottom=0.07)
ax.set_xticks([]) 
ax.set_yticks([]) 

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['left'].set_position(('axes', 0))

ax.set_xbound(0, 1)
ax.set_ybound(0, 1)
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)

ax.set_xlabel("Objective 1", fontsize=18)
ax.set_ylabel("Objective 2", fontsize=18)

arrowed_spines(ax)

# ax.scatter(0.8*np.cos(np.pi/3), 0.8*np.sin(np.pi/3), color='royalblue', s=150, linewidths=2, facecolors='royalblue',
#            edgecolors='white', zorder=10, label="Sampled Solution")

x = np.linspace(0.2, 0.98, 1000)
def superellipse(x, n):
    return (1 - x**n)**(1/n)
y = np.array([superellipse(xi, 3) for xi in x])

# front_points_x = np.array([0.2, 0.5, superellipse(0.8, 3), superellipse(0.5, 3)])
# front_points_y = np.array([superellipse(xi, 3) for xi in front_points_x])

ax.plot(x, y, color='black')

ax.scatter(0.943,superellipse(0.943, 3), color='royalblue', s=300, linewidths=2, facecolors='royalblue',
           edgecolors='white', zorder=10, label="Predicted maximum RTG")

ax.scatter(0.6, 0.7, color='royalblue', s=300, linewidths=2, facecolors='orange',
           edgecolors='white', zorder=10, label="Sampled Solution")

annotate = ax.annotate("", xy=(0.6, 0.7), xytext=(0.943,superellipse(0.943, 3)),
            arrowprops=dict(arrowstyle="<|-|>",
                            edgecolor='crimson',
                            facecolor='crimson',
                            lw=2,
                            shrinkA=2,
                            shrinkB=2),zorder=11, label="Return Deviation")

annotate1 = ax.annotate("", xy=(1.2*np.cos(np.pi/6), 1.2*np.sin(np.pi/6)), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>",
                            edgecolor='royalblue',
                            facecolor='royalblue',
                            lw=2,), label="Condition Preference")

# annotate2 = ax.annotate("", xy=(1.2*np.cos(np.pi/3), 1.2*np.sin(np.pi/3)), xytext=(0, 0),
#             arrowprops=dict(arrowstyle="-|>",
#                             edgecolor='royalblue',
#                             facecolor='royalblue',
#                             lw=2,), label="Sampled Preference")

# annotate3 = ax.annotate("", xy=(0.4*np.cos(np.pi/3), 0.4*np.sin(np.pi/3)), xytext=(0.4*np.cos(np.pi/6), 0.4*np.sin(np.pi/6)),
#             arrowprops=dict(arrowstyle="<|-|>",
#                             connectionstyle="arc3,rad=0.3",
#                             edgecolor='crimson',
#                             facecolor='crimson',
#                             lw=2,), label="Consistency")
# ax.scatter(front_points_x, front_points_y, color='royalblue', s=100)

# h, l = ax.get_legend_handles_labels()
# fig.legend(handles = h + [annotate2, annotate3, annotate], 
#           handler_map={type(annotate) : AnnotationHandler(5)},
#           loc=1)
# plt.savefig("plots/Consistency.pdf")

# ---------------------- WR ---------------------- #
# fig, ax = plt.subplots(1, 1, figsize=(4, 4.5))
# ax.set_xticks([]) 
# ax.set_yticks([]) 

# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.spines['left'].set_position(('axes', 0))

# ax.set_xbound(0, 1)
# ax.set_ybound(0, 1)
# ax.set_xlim(0, 1.05)
# ax.set_ylim(0, 1.125)

# ax.set_xlabel("Objective 1")
# ax.set_ylabel("Objective 2")

# arrowed_spines(ax)

# ax.scatter(0.8*np.cos(np.pi/3), 0.8*np.sin(np.pi/3), color='royalblue', s=150, linewidths=2, facecolors='royalblue',
#            edgecolors='white', zorder=10, label="Sampled Solution")

# annotate = ax.annotate("", xy=(np.cos(np.pi/6), np.sin(np.pi/6)), xytext=(0, 0),
#             arrowprops=dict(arrowstyle="-|>",
#                             edgecolor='black',
#                             facecolor='black',
#                             lw=2,), label="Condition Preference")

# annotate2 = ax.annotate("", xy=(np.cos(np.pi/3), np.sin(np.pi/3)), xytext=(0, 0),
#             arrowprops=dict(arrowstyle="-|>",
#                             edgecolor='royalblue',
#                             facecolor='royalblue',
#                             lw=2,), label="Sampled Preference")

# annotate4 = ax.annotate("", xy=(0, 0), xytext=(0.4*np.sqrt(3)*np.cos(np.pi/6), 0.4*np.sqrt(3)*np.sin(np.pi/6)),
#             arrowprops=dict(arrowstyle="|-|",
#                             edgecolor='crimson',
#                             facecolor='crimson',
#                             shrinkA=0,
#                             shrinkB=0,
#                             lw=3,), label="Weighted Return")

# ax.annotate("", xy=(0.8*np.cos(np.pi/3), 0.8*np.sin(np.pi/3)), xytext=(0.4*np.sqrt(3)*np.cos(np.pi/6), 0.4*np.sqrt(3)*np.sin(np.pi/6)),
#             arrowprops=dict(arrowstyle="-",
#                             ls="--",
#                             edgecolor='crimson',
#                             facecolor='crimson',
#                             lw=1,))
# ax.annotate("", xy=(0.4*np.sqrt(3)*np.cos(np.pi/6)-0.8*0.1414*np.cos(np.pi/12), 0.4*np.sqrt(3)*np.sin(np.pi/6)+0.8*0.1414*np.sin(np.pi/12)),
#                 xytext=(0.4*np.sqrt(3)*np.cos(np.pi/6)-0.8*0.1*np.sin(np.pi/6), 0.4*np.sqrt(3)*np.sin(np.pi/6)+0.8*0.1*np.cos(np.pi/6)),
#             arrowprops=dict(arrowstyle="-",
#                             edgecolor='crimson',
#                             facecolor='crimson',
#                             shrinkA=0,
#                             shrinkB=0,
#                             lw=2,))
# ax.annotate("", xy=(0.4*np.sqrt(3)*np.cos(np.pi/6)-0.8*0.1414*np.cos(np.pi/12), 0.4*np.sqrt(3)*np.sin(np.pi/6)+0.8*0.1414*np.sin(np.pi/12)),
#                 xytext=(0.4*np.sqrt(3)*np.cos(np.pi/6)-0.8*0.1*np.cos(np.pi/6), 0.4*np.sqrt(3)*np.sin(np.pi/6)-0.8*0.1*np.sin(np.pi/6)),
#             arrowprops=dict(arrowstyle="-",
#                             edgecolor='crimson',
#                             facecolor='crimson',
#                             shrinkA=0,
#                             shrinkB=0,
#                             lw=2,))
# ax.scatter(front_points_x, front_points_y, color='royalblue', s=100)

h, l = ax.get_legend_handles_labels()
fig.legend(handles = h + [annotate1, annotate], 
          handler_map={type(annotate) : AnnotationHandler(5)},
          loc=1,
          prop={'size':'15'})
plt.savefig("plots/RD_new.pdf")