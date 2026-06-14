



select balance.total_balance as total_balance, account.account_id, account.name account_name,
account_type.account_type_id, account_type.name account_type_name, balance.month_id, (balance.month_id div 100) as year_id,
link.group_id
from inv_balance_sheet balance, inv_account account, inv_month imonth, inv_account_type account_type, inv_user_group_link link
where balance.account_id = account.account_id
and balance.month_id = imonth.month_id
and account.account_type_id = account_type.account_type_id
and account.user_id = link.user_id
and balance.month_id div 100 > 2007
and balance.month_id div 100 < year(sysdate())
and balance.month_id % 100 = 12
order by link.group_id, account_type.name, account.name, year_id;


select ba.total_balance, acc.account_type_id, floor(ba.month_id / 100) as acc_year, us.name as name
from inv_balance_sheet ba, inv_account acc, inv_user us, inv_user_group_link link
where ba.account_id = acc.account_id 
and acc.user_id = us.user_id
and acc.user_id = link.user_id
and link.group_id = 2
and (ba.month_id % 100) = 12
and floor(ba.month_id / 100) >= 2025
order by acc_year asc;



select
    concat('$', format(sum(balance.total_balance), 2)) as total_balance,
    concat('$', format(sum(balance.cash_balance) + sum(balance.cd_balance), 2)) as cash_balance,
    concat('$', format(sum(balance.money_market), 2)) as semi_balance,
    concat('$', format(sum(balance.total_balance) - sum(balance.money_market) - sum(balance.cash_balance) - sum(balance.cd_balance), 2)) as invest_balance,
    concat(format(100 * (sum(balance.cash_balance) + sum(balance.cd_balance)) / sum(balance.total_balance), 2), '%') as cash_per,
    concat(format(100 * (sum(balance.money_market)) / sum(balance.total_balance), 2), '%') as semi_per,
    concat(format(100 * (sum(balance.total_balance) - sum(balance.money_market) - sum(balance.cash_balance) - sum(balance.cd_balance)) / sum(balance.total_balance), 2), '%') as invest_per,
    account.user_id,
    balance.month_id,
    (balance.month_id div 100) as year_id,
    link.group_id
from inv_balance_sheet balance,
     inv_account account,
     inv_month imonth,
     inv_user_group_link link
where balance.account_id = account.account_id
  and balance.month_id = imonth.month_id
  and account.user_id = link.user_id
  and balance.month_id div 100 > 2020
  and balance.month_id div 100 < year(sysdate())
  and balance.month_id % 100 = 12
  and account.user_id = 2
group by
    account.user_id,
    balance.month_id,
    (balance.month_id div 100),
    link.group_id
order by
    link.group_id,
    year_id;


