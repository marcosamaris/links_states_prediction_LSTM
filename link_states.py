import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from historical.readData.estimateData import read
from historical.readData.estimateData import search_travels
from historical.readData.estimateData import estimate
from historical.readData.estimateData import stops_distance
from historical.readData.travels import haversine2

import matplotlib.pyplot as plt



# if __name__ == '__main__':

pth_files_GTFS = "./historical/readData/"
pth_img_bussim = "./images/bussim/"

trips = pd.read_csv(pth_files_GTFS + 'trips.txt', sep=',')
shapes = pd.read_csv(pth_files_GTFS + 'shapes.txt', sep=',')
stops = pd.read_csv(pth_files_GTFS + 'stops.txt', sep=',')
stopid = pd.read_csv(pth_files_GTFS + 'stop_times.txt', sep=',')   


df, reps = read(pth_files_GTFS + "trips_8700-10-1.dsk", pth_files_GTFS + "interps_8700-10-1.rep")

periods = ['morning', 'm_peak', 'i_peak', 'a_peak', 'night']
#periods = ['morning']


for period in periods:
    travels = search_travels(df, period, 0)
    
    p, mp = stops_distance('8700-10-1', trips, shapes, stops, stopid)
    est_mp = estimate(mp, reps, travels)
    est_p = estimate(p, reps, travels)
    
    segtime_mp = pd.DataFrame(columns=range(len(mp) - 1))
    segtime_p = pd.DataFrame(columns=range(len(p) - 1))
    
    for tr in travels:
        tempo = [a[0] if a.size>0 else np.nan for a in est_mp[tr]]
        row = [tuple([tempo[i+1]-pos for i, pos in enumerate(tempo[:-1])])]
        d = pd.DataFrame(row, columns=range(len(mp)-1))
        segtime_mp = segtime_mp.append(d,ignore_index=True) 
        
        tempo = [a[0] if a.size>0 else np.nan for a in est_p[tr]]
        row = [tuple([tempo[i+1]-pos for i, pos in enumerate(tempo[:-1])])]
        d = pd.DataFrame(row, columns=range(len(mp)-1))
        segtime_p = segtime_p.append(d,ignore_index=True) 
        
        

    meansum_mp = segtime_mp.mean()
    error_mp = segtime_mp.std()
    min_sum_mp = segtime_mp.min()
    max_sum_mp = segtime_mp.max()
    
    meansum_p = segtime_p.mean()
    error_p = segtime_p.std()
    min_sum_p = segtime_p.min()
    max_sum_p = segtime_p.max()
    

    index = np.arange(segtime_mp.shape[1]) 

    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    
    bar_width = 0.35
    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(index, meansum_mp, bar_width,
                     alpha=opacity, color='blue',                     
                     yerr=error_mp,
                     label='MP')
 
    rects2 = plt.bar(index - bar_width, meansum_p, bar_width,
                     alpha=opacity, color='green',
                     yerr=error_p,
                     label='P')
    
    rects3 = plt.plot(index, meansum_mp, bar_width, color='black')
    rects4 = plt.plot(index, meansum_p, bar_width, color="red") 
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Stop Buses', fontsize=20)
    plt.ylabel('Time (Min) in period: ' + period, fontsize=20)
    plt.title('Modeling travel time spent by stop buses - Line 8700-10-1', fontsize=20)
    plt.legend(fontsize=30)
    

    fig.tight_layout()
    fig.savefig(pth_img_bussim + 'barErr_lines_' + period + '-travel-87-10-1.png', dpi=100)
    plt.close(fig) 


    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    
    bar_width = 0.35
    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(index, meansum_mp, bar_width,
                     alpha=opacity, color='blue',                     
                     yerr=error_mp,
                     label='MP')
 
    rects2 = plt.bar(index - bar_width, meansum_p, bar_width,
                     alpha=opacity, color='green',
                     yerr=error_p,
                     label='P')
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Stop Buses', fontsize=20)
    plt.ylabel('Time (Min) in period: ' + period, fontsize=20)
    plt.title('Modeling travel time spent by stop buses - Line 8700-10-1', fontsize=20)
    plt.legend(fontsize=30)
    

    
    df_1 = pd.DataFrame({'MP':meansum_mp,'P':meansum_p})
    lines = df_1.plot.line(title= 'Period: ' + period + ' Line:travel-87-10-1')
    
    fig = lines.get_figure()
    fig.set_size_inches(10, 6)
    fig.savefig(pth_img_bussim + 'lines_' + period + '-travel-87-10-1.png', dpi=100)
    plt.close(fig) 

