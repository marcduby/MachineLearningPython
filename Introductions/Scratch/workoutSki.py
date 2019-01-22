
# imports
import mysql.connector
import matplotlib
from mysql.connector import errorcode
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



# connect to the db
# client = MySQLdb.connect(host="localhost", port=3306, user="root", passwd="yoyoma", db="workout_test")

try:
    connection = mysql.connector.connect(
        user = 'root',
        password = 'yoyoma',
        host = 'localhost',
        database = 'workout_test')

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
    count(distinct wday.day_id) as ski_days, sum(activity.x_ski)/count(distinct wday.day_id) as day_avg, wyear.name_text
from wkt_workout workout, wkt_activity activity, wkt_day wday, wkt_week week, wkt_period period, wkt_year wyear
where workout.day_id = wday.day_id
    and activity.workout_id = workout.workout_id
    and wday.week_id = week.week_id
    and week.period_id = period.period_id
    and period.year_id = wyear.year_id
    and activity.x_ski > 0
group by wyear.name_text
order by xski_total desc, ski_workouts desc
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
        xski_total = row[0]
        xski_workout = row[1]

        print "kms: {}, workouts {}".format(xski_total, xski_workout)


except mysql.connector.Error as err:
    print("Something went wrong: {}".format(err))

finally:
    cursor.close()
    connection.close()




