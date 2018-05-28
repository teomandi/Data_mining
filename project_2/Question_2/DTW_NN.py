import pandas as pd
from ast import literal_eval
import numpy as np
import gmplot
from dtw import dtw
import time


import sys
sys.path.insert(0, '..')
from HarversineDistance import harversine

# reads the datasets
trainSet = pd.read_csv('../../datasets/project_2/train_set.csv', converters={"Trajectory": literal_eval})
test = pd.read_csv('../../datasets/project_2/test_set_a1.csv', sep="\t", converters={"Trajectory": literal_eval})
#trainSet = trainSet[:500]

q_index = 1
# for every query in test we find the closest journeys
for query in test['Trajectory']:
    query = np.array(query)

    # plot the query in goolge map
    lats = []
    lons = []
    for spot in query:
        lats.append(spot[2])
        lons.append(spot[1])

    gmap = gmplot.GoogleMapPlotter(lats[int(len(lats)/2)], lons[int(len(lons)/2)], 11)
    gmap.plot(lats, lons, 'cornflowerblue', edge_width=5)
    gmap.draw("Q2A1_produced/Q2_query_"+str(q_index)+".html")

    distances = []
    index = 0
    start_time = time.time()

    # callcates the distances of the query and every single journey in trainSet
    for journey in trainSet['Trajectory']:
        journey = np.array(journey)
        distances.append((dtw(query, journey, dist=lambda spot1, spot2: harversine
                (spot1[1], spot1[2], spot2[1], spot2[2]))[0], index))
        index += 1
    # sorts the distances and plots the top 5
    distances = sorted(distances)
    elapsed_time = time.time() - start_time
    for i in range(5):
        print(distances[i][0], trainSet['journeyPatternId'][distances[i][1]])

        lats = []
        lons = []
        for spot in trainSet['Trajectory'][distances[i][1]]:
            lats.append(spot[2])
            lons.append(spot[1])
        gmap = gmplot.GoogleMapPlotter(lats[int(len(lats)/2)], lons[int(len(lons)/2)], 11)
        gmap.plot(lats, lons, 'cornflowerblue', edge_width=5)
        gmap.draw("Q2A1_produced/Q2_"+str(distances[i][1])+".html")

    # creates the requested page..we use iframes to show the maps..
    f = open('Q2A1_produced/finals/final_'+str(q_index)+'.html', 'w')
    message = """
<!DOCTYPE html>
<html>
<head>

<style>
.center {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 50%;
}
</style>
</style>
</head>
<body>
<br/><br/><br/><br/><br/><br/><br/>

<table  class = 'center'>
  <tr>
    <td><iframe src="../Q2_query_"""+str(q_index)+""".html"></iframe></td>
    <td><iframe src="../Q2_"""+str(distances[0][1])+""".html"></iframe></td>
    <td><iframe src="../Q2_"""+str(distances[1][1])+""".html"></iframe></td>
  </tr>
  <tr>
    <td>Test Trip """+str(q_index)+"""</td>
    <td>Neighbor 1</td>
    <td>Neighbor 2</td>
  </tr>
  <tr>
    <td>Δt= """+str(elapsed_time)+"""sec</td>
    <td>JP_ID: """+trainSet['journeyPatternId'][distances[0][1]]+"""</td>
    <td>JP_ID: """+trainSet['journeyPatternId'][distances[1][1]]+"""</td>
  </tr>
  <tr>
    <td></td>
    <td>DTW:  """+str(distances[0][0])+"""</td>
    <td>DTW:  """+str(distances[1][0])+"""</td>
  </tr>
   <tr>
    <td><iframe src="../Q2_"""+str(distances[2][1])+""".html"></iframe></td>
    <td><iframe src="../Q2_"""+str(distances[3][1])+""".html"></iframe></td>
    <td><iframe src="../Q2_"""+str(distances[4][1])+""".html"></iframe></td>
  </tr>
  <tr>
    <td>Neighbor 3</td>
    <td>Neighbor 4</td>
    <td>Neighbor 5</td>
  </tr>
  <tr>
    <td>JP_ID: """+trainSet['journeyPatternId'][distances[2][1]]+"""</td>
    <td>JP_ID: """+trainSet['journeyPatternId'][distances[3][1]]+"""</td>
    <td>JP_ID: """+trainSet['journeyPatternId'][distances[4][1]]+"""</td>
  </tr>
  <tr>
    <td>DTW: """+str(distances[2][0])+"""</td>
    <td>DTW: """+str(distances[3][0])+"""</td>
    <td>DTW: """+str(distances[4][0])+"""</td>
  </tr>
</table>

</body>
</html>"""

    f.write(message)
    f.close()
    q_index += 1
    print("\n\n\n")
