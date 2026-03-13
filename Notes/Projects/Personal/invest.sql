



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




