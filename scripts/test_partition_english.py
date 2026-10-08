"""English counterpart of the 32-message/20-question Chinese fixture."""
from scripts import test_partition_types as benchmark


FACTS = [
    'Lin\'s favorite football team is Arsenal.',
    'Lin is allergic to peanuts and cannot eat foods containing peanuts.',
    'Zhou works at Riverside School as a music teacher.',
    'Mei likes unsweetened soy milk and dislikes sweetened soy milk.',
    'Lin\'s older sister is Mei.',
    'Mei\'s teacher is Zhou.',
    'Wang and Lin are colleagues.',
    'Chen is Wang\'s father.',
    'When writing weekly reports for Lin, put the conclusion first, then list progress in bullet points.',
    'Only urgent incident reports may omit the background introduction; all other reports must include it.',
    'When exporting spreadsheets for Mei, dates must use the YYYY-MM-DD format.',
    'Before publishing an official document, Wang must review it first, then Lin must approve it.',
    'Lin participated in the Hangzhou Marathon in October 2023.',
    'Mei went to Beijing for a concert on May 3, 2024.',
    'Wang originally planned a business trip to Shanghai in June 2024, but later cancelled it.',
    'Chen completed training in February 2024, then passed the exam in March 2024.',
    'Lin used to live in Hangzhou, moved to Shanghai in March 2024, and now lives in Shanghai.',
    'Zhou teaches violin, not piano.',
    'Wang\'s favorite team is Manchester United.',
    'Lin watched a Chelsea match yesterday, but he does not support Chelsea.',
    'Mei bought a Tottenham shirt as a birthday gift for Wang.',
    'Wang likes sweetened soy milk and dislikes unsweetened soy milk.',
    'Chen works at Mountain School, teaching mathematics.',
    'Zhang\'s teacher is Li.',
    'When writing weekly reports for Wang, start with the background and use continuous paragraphs.',
    'When exporting spreadsheets for Chen, dates use the DD/MM/YYYY format.',
    'Wang participated in the Shanghai Marathon in 2022.',
    'Lin plans to travel to Beijing on business next year, but it is not confirmed.',
    'Mei used to live in Shanghai and now lives in Suzhou.',
    'Chen is allergic to shrimp but not to peanuts.',
    'Ordinary reports must retain the background introduction; the urgent incident report exception does not apply.',
    'Zhang\'s colleague is Liu.',
]
QUERIES = [
    'What is Lin\'s favorite football team?',
    'What food is Lin allergic to?',
    'Where does Zhou work?',
    'What kind of soy milk does Mei like?',
    'What is the name of Lin\'s older sister?',
    'Who is Mei\'s teacher?',
    'What is the relationship between Wang and Lin?',
    'Who is Wang\'s father?',
    'How should Lin\'s weekly reports be organized?',
    'Under what circumstances may a report omit the background introduction?',
    'What date format should be used when exporting spreadsheets for Mei?',
    'In what order must an official document be reviewed and approved before publication?',
    'What race did Lin participate in during October 2023?',
    'On what date did Mei go to Beijing for a concert?',
    'What happened to Wang\'s planned June business trip to Shanghai?',
    'In what order did Chen complete training and pass the exam?',
    'Where does Lin live after moving, and where did he live before?',
    'What instrument does Lin\'s older sister\'s teacher teach?',
    'What kind of soy milk does Lin\'s older sister like?',
    'Where does Wang\'s father work?',
]


if __name__ == '__main__':
    assert len(FACTS)==len(benchmark.FACTS) and len(QUERIES)==len(benchmark.CASES)
    benchmark.FACTS=FACTS
    benchmark.CASES=[(category,query,gold) for query,(category,_,gold) in zip(QUERIES,benchmark.CASES)]
    benchmark.main()
