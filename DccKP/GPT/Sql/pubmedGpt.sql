

-- pull abstracts for a gene
select abs.abstract_text 
from pmd_abstract abs, pmd_keyword keyword, pmd_link_keyword_abstract link 
where link.abstract_id = abs.id and link.keyword_id = keyword.id 
and keyword.keyword = 'PPARG';

select abs.abstract_text 
from pmd_abstract abs, pmd_keyword keyword, pmd_link_keyword_abstract link 
where link.abstract_id = abs.id and link.keyword_id = keyword.id 
and lower(keyword.keyword) = 'pparg';

select abs.abstract_text 
from pmd_abstract abs
where abs.abstract_text like '%PPARG%';





select abs.abstract_text 
from pmd_abstract abs, pmd_keyword keyword, pmd_link_keyword_abstract link 
where link.abstract_id = abs.id and link.keyword_id = keyword.id 
and lower(keyword.keyword) = 'slc30a8';


select abs.abstract_text 
from pmd_abstract abs
where abs.abstract_text like '%SLC30A8%';

, pmd_keyword keyword, pmd_link_keyword_abstract link 
where link.abstract_id = abs.id and link.keyword_id = keyword.id 
and lower(keyword.keyword) = 'slc30a8';



