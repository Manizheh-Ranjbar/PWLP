#feature 2: real nodefeature 3:without node feature
#Cora pca 10, Citeseer pca 50, pubmed pca 15
foreach ($db in "EML", "ADV" , "Router", "KHN", "HPD", "SMG", "ZWL","Cora") {
        python Main.py --data-name $db --test-ratio=0.2 --feature 3 --pca 10 --seed 1
}


foreach ($db in "Cora", "ADV", "EML" , "Router", "KHN", "HPD", "SMG", "ZWL") {
    python Main.py --data-name $db --test-ratio=0.2 --feature 3 --pca 10
    python Main.py --data-name $db --test-ratio=0.2 --feature 3 --pca 10 --mask
}

python Main.py --data-name $db --test-ratio=0.2 --feature 2 --pca 10
python Main.py --data-name $db --test-ratio=0.2 --feature 2 --pca 10 --mask

# foreach ($db in "Cora", "Citeseer" , "Pubmed") {
foreach ($seed in 2,3,4,5) {
    foreach ($feature in 2,3) {
        python Main.py --data-name Cora --test-ratio=0.2 --feature $feature --pca 10 --seed $seed
        python Main.py --data-name Citeseer --test-ratio=0.2 --feature $feature --pca 50 --seed $seed
        python Main.py --data-name Pubmed --test-ratio=0.2 --feature $feature --pca 15 --seed $seed
    }
}
        python Main.py --data-name Pubmed --test-ratio=0.2 --feature $feature --pca 5 --seed $seed ??????


    foreach ($feature in 2,3) {
        python Main.py --data-name Cora --test-ratio=0.2 --feature $feature --seed 1
        python Main.py --data-name Citeseer --test-ratio=0.2 --feature $feature --seed 1
        python Main.py --data-name Pubmed --test-ratio=0.2 --feature $feature --seed 1
    }






foreach ($seed in 3,4,5) {
    python Main.py --data-name Cora --test-ratio=0.2 --feature 3 --pca 10 --seed $seed
}
foreach ($seed in 1,2 ,3,4,5) {
    python Main.py --data-name Cora --test-ratio=0.2 --feature 2 --pca 10 --seed $seed
    python Main.py --data-name Citeseer --test-ratio=0.2 --feature 3 --pca 50 --seed $seed
    python Main.py --data-name Pubmed --test-ratio=0.2 --feature 3 --pca 15 --seed $seed
    python Main.py --data-name Citeseer --test-ratio=0.2 --feature 2 --pca 50 --seed $seed
    python Main.py --data-name Pubmed --test-ratio=0.2 --feature 2 --pca 15 --seed $seed
}
foreach ($seed in 1,2 ) {
    python Main.py --data-name Pubmed --test-ratio=0.2 --feature 3 --pca 15 --seed $seed
    python Main.py --data-name Pubmed --test-ratio=0.2 --feature 2 --pca 15 --seed $seed
}
