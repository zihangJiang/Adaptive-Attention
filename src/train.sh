if [ ! $2 ];
then
    gpu="0"
else
	gpu="$2"
fi
echo "using gpu" $gpu
if [ $1 == "Mini" ];
then
	echo "training MiniImageNet experiments";
	python train.py -exp ../exp/MiniImageNet/5way/1shot/unweighted -g $gpu -up -uic 
	python train.py -exp ../exp/MiniImageNet/5way/5shot/unweighted -g $gpu -up -uic  -nsVa 5 -nsTr 5


elif [ $1 == "Omni" ];
	then
	echo "training Omniglot experiments";
	python train.py -data Omniglot -g $gpu -exp ../exp/Omniglot/5way/1shot/unweighted -up -uic -cTr 60 -btz 600
	python train.py -data Omniglot -g $gpu -exp ../exp/Omniglot/5way/5shot/unweighted -up -uic  -nsVa 5 -nsTr 5

elif [ $1 == "CUB" ]; 
	then
	echo "training CUB experiments";
	python train.py -data CUB -g $gpu -nep 100 -lrS 20 -lrG 0.5 -exp ../exp/CUB/5way/1shot/unweighted -up -uic 
	python train.py -data CUB -g $gpu -nep 100 -lrS 20 -lrG 0.5 -exp ../exp/CUB/5way/5shot/unweighted -cTr 15 -btz 225 -up -uic  -nsVa 5 -nsTr 5

elif [ $1 == "Dog" ]; 
	then
	echo "training StanfordDog experiments";
	python train.py -data StanfordDog -g $gpu -nep 100 -lrS 20 -lrG 0.5 -exp ../exp/StanfordDog/5way/1shot/unweighted -up -uic 
	python train.py -data StanfordDog -g $gpu -nep 100 -lrS 20 -lrG 0.5 -exp ../exp/StanfordDog/5way/5shot/unweighted -cTr 15 -btz 225 -up -uic  -nsVa 5 -nsTr 5

elif [ $1 == "Car" ]; 
	then
	echo "training StanfordCar experiments";
	python train.py -data StanfordCar -g $gpu -nep 100 -lrS 20 -lrG 0.5 -exp ../exp/StanfordCar/5way/1shot/unweighted -up -uic 
	python train.py -data StanfordCar -g $gpu -nep 100 -lrS 20 -lrG 0.5 -exp ../exp/StanfordCar/5way/5shot/unweighted -cTr 15 -btz 225 -up -uic  -nsVa 5 -nsTr 5
fi
