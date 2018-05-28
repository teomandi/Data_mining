import pandas as pd
from ast import literal_eval
import numpy as np
import gmplot
import time
import sys
from LCSS import LonsLatsLCS, LCS

trainSet = pd.read_csv('../../datasets/project_2/train_set.csv', converters={"Trajectory": literal_eval})
test = pd.read_csv('../../datasets/project_2/test_set_a2.csv', sep="\t", converters={"Trajectory": literal_eval})
trainSet = trainSet[:1000]

q_index = 1
for query in test['Trajectory']:
    query = np.array(query)	
    lats = []
    lons = []
    for spot in query:
        lats.append(spot[2])
        lons.append(spot[1])
    gmap = gmplot.GoogleMapPlotter(lats[int(len(lats)/2)], lons[int(len(lons)/2)], 11)
    gmap.plot(lats, lons, 'green', edge_width=5)
    gmap.draw("Q2A2_produced/Q2_2query_"+str(q_index)+".html")

    results = []
    index = 0
    start_time = time.time()

    # query oi diadromes apo to test
    # journey oi diadromes apo to train
    for journey in trainSet['Trajectory']:
        journey = np.array(journey)
        infoLCS = LonsLatsLCS(query, journey)
        results.append([infoLCS, index])
        index =index+1

    # find the 5 journeys with the most commons
    maxFive = []
    maxIndex = []
    for i in range(5):
        maxFound=[]
        max_v = 0
        for res in results:
            if max_v < res[0][0] and (res[1] not in maxIndex):
                maxFound = res
                max_v = res[0][0]
        maxFive.append(maxFound)
        if len(maxFound) != 0:
            maxIndex.append(maxFound[1])

    elapsed_time = time.time() - start_time
    for i in range(5):
        lats = []
        lons = []
        for spot in trainSet['Trajectory'][maxFive[i][1]]:
            lats.append(spot[2])
            lons.append(spot[1])

        commonLats=[]
        commonLons=[]
        for spot in maxFive[i][0][1]:
            commonLats.append(spot[2])
            commonLons.append(spot[1])

        gmap = gmplot.GoogleMapPlotter(lats[int(len(lats)/2)], lons[int(len(lons)/2)], 11)
        gmap.plot(lats, lons, 'green', edge_width=5)
        gmap.plot(commonLats, commonLons, 'red', edge_width=5)
        gmap.draw("Q2A2_produced/Q2_2q"+str(q_index)+"id" +str(maxFive[i][1])+".html", )

    for i in range(5):
        print("input file name: Q2_2q", q_index, "id", maxFive[i][1])
        print("matching points: ", maxFive[i][0][0])
        print("id: ", trainSet['journeyPatternId'][maxFive[i][1]])
    print("------------------------------------------------------------------")

    f = open('Q2A2_produced/finals/final'+str(q_index)+'.html', 'w')
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
    <td><iframe src="../Q2_2query_"""+str(q_index)+""".html"></iframe></td>
    <td><iframe src="../Q2_2q"""+str(q_index)+"""id"""+str(maxFive[0][1])+""".html"></iframe></td>
    <td><iframe src="../Q2_2q"""+str(q_index)+"""id"""+str(maxFive[1][1])+""".html"></iframe></td>
  </tr>
  <tr>
    <td>Test Trip """+str(q_index)+"""</td>
    <td>Neighbor 1</td>
    <td>Neighbor 2</td>
  </tr>
  <tr>
    <td>Δt= """+ str(elapsed_time)+"""sec</td>
    <td>JP_ID: """+str(trainSet['journeyPatternId'][maxFive[0][1]])+"""</td>
    <td>JP_ID: """+str(trainSet['journeyPatternId'][maxFive[1][1]])+"""</td>
  </tr>
  <tr>
    <td></td>
    <td>#Matching Points:  """+str(maxFive[0][0][0])+"""</td>
    <td>#Matching Points:  """+str(maxFive[1][0][0])+"""</td>
  </tr>
   <tr>
    <td><iframe src="../Q2_2q"""+str(q_index)+"""id"""+str(maxFive[2][1])+""".html"></iframe></td>
    <td><iframe src="../Q2_2q"""+str(q_index)+"""id"""+str(maxFive[3][1])+""".html"></iframe></td>
    <td><iframe src="../Q2_2q"""+str(q_index)+"""id"""+str(maxFive[4][1])+""".html"></iframe></td>
  </tr>
  <tr>
    <td>Neighbor 3</td>
    <td>Neighbor 4</td>
    <td>Neighbor 5</td>
  </tr>
  <tr>
    <td>JP_ID: """+str(trainSet['journeyPatternId'][maxFive[2][1]])+"""</td>
    <td>JP_ID: """+str(trainSet['journeyPatternId'][maxFive[3][1]])+"""</td>
    <td>JP_ID: """+str(trainSet['journeyPatternId'][maxFive[4][1]])+"""</td>
  </tr>
  <tr>
    <td>#Matching Points:"""+str(maxFive[2][0][0])+"""</td>
    <td>#Matching Points:"""+str(maxFive[3][0][0])+"""</td>
    <td>#Matching Points:"""+str(maxFive[4][0][0])+"""</td>
  </tr>
</table>

</body>
</html>"""

    f.write(message)
    f.close()
    q_index += 1
