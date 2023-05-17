mkdir data
cd data
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar - xvzf aclImdb_v1.tar.gz
cd ..
cd model
mkdir weights
cd weights
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1MGHfrFdwdzPiTib20NnOBkEGi0EdXMxs' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1MGHfrFdwdzPiTib20NnOBkEGi0EdXMxs" -O tmd_ckpts.zip && rm -rf /tmp/cookies.txt
unzip tmd_ckpts.zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1oRKihueWcrh0okSB3fPWuLZKEyq1hwCL' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1oRKihueWcrh0okSB3fPWuLZKEyq1hwCL" -O tmd_ckpts_roberta.zip && rm -rf /tmp/cookies.txt
unzip tmd_ckpts_roberta.zip
mkdir agnews
cd agnews
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1iyulolV8Z4jrePwJHB6L0UwXmH_vjdMy' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1iyulolV8Z4jrePwJHB6L0UwXmH_vjdMy" -O mask-len128-epo10-batch32-rate0.9-best.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1F8nck3G7CVhQEfuBZZcwbECAG4HPofWH' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1F8nck3G7CVhQEfuBZZcwbECAG4HPofWH" -O roberta_mask-len128-epo10-batch32-rate0.9-best.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1X1wewPtBQDlTSmdAanUpGY096E1SIuAF' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1X1wewPtBQDlTSmdAanUpGY096E1SIuAF" -O perturbation_constraint_pca0.8_100.pkl && rm -rf /tmp/cookies.txt
cd ..
mkdir imdb
cd imdb
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1yIS1R86YtUeWSlLvuEM4QDuN61cXPhS3' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1yIS1R86YtUeWSlLvuEM4QDuN61cXPhS3" -O mask-len256-epo10-batch32-rate0.3-best.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1q5kGXU5aNAhDRliWyO1sVNYCagh-pYlB' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1q5kGXU5aNAhDRliWyO1sVNYCagh-pYlB" -O roberta_mask-len256-epo10-batch32-rate0.3-best.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1p8XUKBqnUFsFBG9QSX0DI5oKFC4lsjTV' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1p8XUKBqnUFsFBG9QSX0DI5oKFC4lsjTV" -O perturbation_constraint_pca0.8_100_imdb.pkl && rm -rf /tmp/cookies.txt
