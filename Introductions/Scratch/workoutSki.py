
# imports
import MySQLdb
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



# connect to the db
client = MySQLdb.connect(host="localhost", port=3306, user="root", passwd="yoyoma", db="workout_test")

try:
    cursor = client.cursor()
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
order by xski_total desc, ski_workouts desc;
    """

    # execute the query
    cursor.execute(query)

    # get the results
    result = cursor.fetchAll()

    # loop though the rows
    for row in results:
        xski_total = row[0]
        xski_workout = row[1]

        print "kms: {}, workouts {}".format(xski_total, xski_workout)


except Exception:
    print("got a mysql error: " + Exception)
finally:
    client.close()




