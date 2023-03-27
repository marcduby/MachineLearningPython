

# https://www.youtube.com/watch?v=xPk7S-Eb4J4

# imports
import pandas as pd 
import folium 

# this makes it so that you see all the columns in a pd.show()
pd.set_option('display.max_columns', None)

# load the data 
# https://github.com/practicalaifab/folium/blob/codespace-practicalaifab-probable-space-umbrella-r4476xv556q356qr/data/hospitals.csv
# https://raw.githubusercontent.com/practicalaifab/folium/codespace-practicalaifab-probable-space-umbrella-r4476xv556q356qr/data/hospitals.csv
df = pd.read_csv("https://raw.githubusercontent.com/practicalaifab/folium/codespace-practicalaifab-probable-space-umbrella-r4476xv556q356qr/data/hospitals.csv")

# get list of hoospitals 

# filter for only MA hospitals 
ma = df[df['STATE'] == 'MA']
ma = ma[['NAME', 'LATITUDE', 'LONGITUDE']]

# display 
# map.head()


# get the mean lat/lon for the map crteation 
lat_mean = ma['LATITUDE'].mean()
lon_mean = ma['LONGITUDE'].mean()

# create folium map 
map = folium.Map(location=[lat_mean, lon_mean], zoom_start=15)

# need to creat list of hospitals to put them on the map 
list_hosp = ma.values.tolist()

# loop over list 
for index in list_hosp:
    # add to map 
    map.add_child(folium.Marker(location=[index[1], index[2]], popup=index[0], icon=folium.Icon(color='green')))


# save map  as html file 
map.save("ma.html")


# df.show()


