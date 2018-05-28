import gmplot
import pandas as pd
from ast import literal_eval

trainSet = pd.read_csv('../../datasets/project_2/train_set.csv',
                       converters={"Trajectory": literal_eval}, index_col='tripId')

journeys = [1, 4, 12, 43, 48]

for journey in journeys:
    lats = []
    lons = []
    for spot in trainSet['Trajectory'][journey]:
        lats.append(spot[2])
        lons.append(spot[1])

    gmap = gmplot.GoogleMapPlotter(lats[0], lons[0], 12)
    gmap.plot(lats[0:], lons[0:], 'cornflowerblue', edge_width=5)

    gmap.draw("Q1_produced/"+trainSet['journeyPatternId'][journey]+".html")
