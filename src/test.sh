if [ ! $2 ];
then
    gpu="0"
else
	gpu="$2"
fi
echo "using gpu" $gpu
if [ ! $3 ];
then
	if [ $1 == "Mini" ];
	then
		echo "training MiniImageNet experiments";
		python test.py -exp ../exp/MiniImageNet/5way/1shot/unweighted -g $gpu -btz 75 -its 600 -up -uic -r
		python test.py -exp ../exp/MiniImageNet/5way/5shot/unweighted -g $gpu -btz 75 -its 600 -up -uic -r -nsVa 5 -nsTr 5


	elif [ $1 == "Omni" ];
		then
		echo "testing Omniglot experiments";
		python test.py -data Omniglot -g $gpu -btz 75 -its 1000 -exp ../exp/Omniglot/5way/1shot/unweighted -up -uic -r
		python test.py -data Omniglot -g $gpu -btz 75 -its 1000 -exp ../exp/Omniglot/5way/5shot/unweighted -up -uic -r -nsVa 5 -nsTr 5

	elif [ $1 == "CUB" ]; 
		then
		echo "testing CUB experiments";
		python test.py -data CUB -g $gpu -btz 75 -its 600 -exp ../exp/CUB/5way/1shot/unweighted -up -uic -r
		python test.py -data CUB -g $gpu -btz 75 -its 600 -exp ../exp/CUB/5way/5shot/unweighted -up -uic -r -nsVa 5 -nsTr 5

	elif [ $1 == "Dog" ]; 
		then
		echo "testing StanfordDog experiments";
		python test.py -data StanfordDog -g $gpu -btz 75 -its 600 -exp ../exp/StanfordDog/5way/1shot/unweighted -up -uic -r
		python test.py -data StanfordDog -g $gpu -btz 75 -its 600 -exp ../exp/StanfordDog/5way/5shot/unweighted -up -uic -r -nsVa 5 -nsTr 5

	elif [ $1 == "Car" ]; 
		then
		echo "testing StanfordCar experiments";
		python test.py -data StanfordCar -g $gpu -btz 75 -its 600 -exp ../exp/StanfordCar/5way/1shot/unweighted -up -uic -r
		python test.py -data StanfordCar -g $gpu -btz 75 -its 600 -exp ../exp/StanfordCar/5way/5shot/unweighted -up -uic -r -nsVa 5 -nsTr 5
	fi
else
	echo "testing" $3 "experiments"
	if [ $1 == "CUB" ]; 
		then
		echo "testing CUB experiments";
		python test.py -data CUB -g $gpu -btz 75 -its 600 -exp ../exp/CUB/5way/1shot/unweighted -proto -r
		python test.py -data CUB -g $gpu -btz 75 -its 600 -exp ../exp/CUB/5way/5shot/unweighted -proto -r -nsVa 5 -nsTr 5

	elif [ $1 == "Dog" ]; 
		then
		echo "testing StanfordDog experiments";
		python test.py -data StanfordDog -g $gpu -btz 75 -its 600 -exp ../exp/StanfordDog/5way/1shot/unweighted -proto -r
		python test.py -data StanfordDog -g $gpu -btz 75 -its 600 -exp ../exp/StanfordDog/5way/5shot/unweighted -proto -r -nsVa 5 -nsTr 5

	elif [ $1 == "Car" ]; 
		then
		echo "testing StanfordCar experiments";
		python test.py -data StanfordCar -g $gpu -btz 75 -its 600 -exp ../exp/StanfordCar/5way/1shot/unweighted -proto -r
		python test.py -data StanfordCar -g $gpu -btz 75 -its 600 -exp ../exp/StanfordCar/5way/5shot/unweighted -proto -r -nsVa 5 -nsTr 5
	fi
fi