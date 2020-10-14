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
	python train.py -exp ../exp/proto/MiniImageNet/5way/1shot/unweighted -g $gpu -proto
	python train.py -exp ../exp/proto/MiniImageNet/5way/5shot/unweighted -g $gpu -proto -nsVa 5 -nsTr 5 -cTr 15 -btz 225

elif [ $1 == "CUB" ]; 
	then
	echo "training CUB experiments";
	python train.py -data CUB -g $gpu -proto -nep 100 -lrS 20 -lrG 0.5 -exp ../exp/proto/CUB/5way/1shot/unweighted 
	python train.py -data CUB -g $gpu -proto -nep 100 -lrS 20 -lrG 0.5 -exp ../exp/proto/CUB/5way/5shot/unweighted -cTr 15 -btz 225 -nsVa 5 -nsTr 5

elif [ $1 == "Dog" ]; 
	then
	echo "training StanfordDog experiments";
	python train.py -data StanfordDog -g $gpu -proto -nep 100 -lrS 20 -lrG 0.5 -exp ../exp/proto/StanfordDog/5way/1shot/unweighted 
	python train.py -data StanfordDog -g $gpu -proto -nep 100 -lrS 20 -lrG 0.5 -exp ../exp/proto/StanfordDog/5way/5shot/unweighted -cTr 15 -btz 225 -nsVa 5 -nsTr 5

elif [ $1 == "Car" ]; 
	then
	echo "training StanfordCar experiments";
	python train.py -data StanfordCar -g $gpu -proto -nep 100 -lrS 20 -lrG 0.5 -exp ../exp/proto/StanfordCar/5way/1shot/unweighted 
	python train.py -data StanfordCar -g $gpu -proto -nep 100 -lrS 20 -lrG 0.5 -exp ../exp/proto/StanfordCar/5way/5shot/unweighted -cTr 15 -btz 225 -nsVa 5 -nsTr 5
fi
