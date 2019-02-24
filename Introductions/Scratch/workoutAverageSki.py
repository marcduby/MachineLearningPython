
# imports
import mysql.connector
import matplotlib
from mysql.connector import errorcode
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# main variables
x = []
x1 = []
x2 = []
x3 = []
yski_avg_wkt = []
znum_wkt = []
pski_avg_year = []

# connect to the db
# client = MySQLdb.connect(host="localhost", port=3306, user="root", passwd="yoyoma", db="workout_test")
try:
    connection = mysql.connector.connect(
        user = 'root',
        password = 'yoyoma',
        host = 'localhost',
        database = 'workout')

    # log
    print("connected to database")

except mysql.connector.Error as error:
    if error.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("invalid credentials")
    elif error.errno == errorcode.ER_BAD_DB_ERROR:
        print("database not found")
    else:
        print(error)

try:
    cursor = connection.cursor(buffered = True)

    # build the query
    query = """
    select sum(activity.x_ski) as xski_total, count(1) as ski_workouts, sum(activity.x_ski)/count(1) as workout_avg,
    count(distinct wday.day_id) as ski_days, sum(activity.x_ski)/count(distinct wday.day_id) as day_avg, 
    sum(activity.x_ski)/dayofyear(sysdate()) as running_avg, wyear.name
from wkt_workout workout, wkt_activity activity, wkt_day wday, wkt_week week, wkt_period period, wkt_year wyear
where workout.day_id = wday.day_id
    and activity.workout_id = workout.workout_id
    and wday.week_id = week.week_id
    and week.period_id = period.period_id
    and period.year_id = wyear.year_id
    and activity.x_ski > 0 
    and (dayofyear(wday.date) <= dayofyear(sysdate()) or year(wday.date) < wyear.name)
group by wyear.name_text
order by wyear.name
    """

    # query = "select sum(activity.x_ski) as xski_total, count(1) as ski_workouts from wkt_activity activity"

    # execute the query
    cursor.execute(query)

    # log
    print("run query")

    # get the results
    results = cursor.fetchall()

    # log
    print("got results")

    # loop though the rows
    for row in results:
        # get ski total
        ski_total = row[0]

        # append the ski avg per day
        xski_avg = row[4]
        yski_avg_wkt.append(xski_avg)

        # append the ski avg per year day
        xski_avg_year_day = row[5]
        pski_avg_year.append(xski_avg_year_day)

        # get number workouts
        xski_workout_numbers = row[3]
        znum_wkt.append(xski_workout_numbers)

        # get year names
        year_name = row[6]
        x.append(year_name)

        # set the positions for the bars
        x1.append(int(year_name) - 0.2)
        x2.append(int(year_name))
        x3.append(int(year_name) + 0.2)

        # log
        print "ski total: {}, minutes per day: {}, minuted for whole year: {}, ski days {}, year name {}".format(ski_total, xski_avg, xski_avg_year_day, xski_workout_numbers, year_name)


except mysql.connector.Error as err:
    print("Something went wrong: {}".format(err))

finally:
    cursor.close()
    connection.close()

# plot
# plt.plot(x, y)
# plt.bar(x, y)

# add in subplots
sub = plt.subplot(111)
sub.bar(x1, yski_avg_wkt, width = 0.2, color = 'g', align = 'center')
sub.bar(x2, znum_wkt, width = 0.2, color = 'r', align = 'center')
sub.bar(x3, pski_avg_year, width = 0.2, color = 'b', align = 'center')

plt.xticks(x)
plt.axes().yaxis.grid()

# log
print('plot created')

# set options
# plt.interactive(False)

# show the plot
plt.show()
print('show plot')



