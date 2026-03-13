

-- total xski workouts and time ortdererd by time
select min(day.date), max(day.date), count(activity.x_ski) as count_workouts, 
	count(distinct day.date) as count_days, sum(activity.x_ski) as total_time, 
	truncate(sum(activity.x_ski)/count(activity.x_ski), 2) as avg_per_workout, 
	truncate(sum(activity.x_ski)/count(distinct day.date), 2) as avg_per_day, 
	year.name, year.year_id
from wkt_activity activity, wkt_day day, wkt_workout workout, 
	wkt_week week, wkt_period period, wkt_year year
where year.year_id = period.year_id
	and period.period_id = week.period_id
	and week.week_id = day.week_id
	and day.day_id = workout.day_id
	and workout.workout_id = activity.workout_id
	and activity.x_ski > 0
group by year.name, year.year_id
order by total_time;

-- total xski workouts and time ordered by yearselect min(day.date), max(day.date), count(activity.x_ski) as count_workouts, 
select min(day.date), max(day.date), count(activity.x_ski) as count_workouts, 
	count(distinct day.date) as count_days, sum(activity.x_ski) as total_time, 
	truncate(sum(activity.x_ski)/count(activity.x_ski), 2) as avg_per_workout, 
	truncate(sum(activity.x_ski)/count(distinct day.date), 2) as avg_per_day, 
	year.name, year.year_id
from wkt_activity activity, wkt_day day, wkt_workout workout, 
	wkt_week week, wkt_period period, wkt_year year
where year.year_id = period.year_id
	and period.period_id = week.period_id
	and week.week_id = day.week_id
	and day.day_id = workout.day_id
	and workout.workout_id = activity.workout_id
	and activity.x_ski > 0
group by year.name, year.year_id
order by year.name;


-- total rowing workouts per calendar year
select min(day.date), max(day.date), count(activity.rowing) as count_workouts, 
	count(distinct day.date) as count_days, sum(activity.rowing) as total_time, 
	truncate(sum(activity.rowing)/count(activity.rowing), 2) as avg_per_workout, 
	truncate(sum(activity.rowing)/count(distinct day.date), 2) as avg_per_day, 
	year(day.date) as year
from wkt_activity activity, wkt_day day, wkt_workout workout
where day.day_id = workout.day_id
	and workout.workout_id = activity.workout_id
	and activity.rowing > 0
group by year
order by year;


-- total swimming workouts per calendar year
select min(day.date), max(day.date), count(activity.swim) as count_workouts, 
	count(distinct day.date) as count_days, sum(activity.swim) as total_time, 
	truncate(sum(activity.swim)/count(activity.swim), 2) as avg_per_workout, 
	truncate(sum(activity.swim)/count(distinct day.date), 2) as avg_per_day, 
	year(day.date) as year
from wkt_activity activity, wkt_day day, wkt_workout workout
where day.day_id = workout.day_id
	and workout.workout_id = activity.workout_id
	and activity.swim > 0
group by year
order by year;


-- total down hill ski by year
select min(day.date), max(day.date), count(activity.activity_id) as total_ski, year.name as year
from wkt_activity activity, wkt_day day, wkt_workout workout, 
	wkt_week week, wkt_period period, wkt_year year
where year.year_id = period.year_id
	and period.period_id = week.period_id
	and week.week_id = day.week_id
	and day.day_id = workout.day_id
	and workout.workout_id = activity.workout_id
    and workout.workout_type_id = 31
group by year.name;



