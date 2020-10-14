if [ $1 == "MiniImageNet" ]; 
then
	echo "training MiniImageNet experiments";
	python train.py -cTr 15 -g 1 -btz 225 -exp ../exp/MiniImageNet/5way15/1shot/unweighted -up -uic 
	python train.py -cTr 15 -g 1 -btz 225 -exp ../exp/MiniImageNet/5way15/1shot/weighted -up -uic -fm weighted
	python train.py -cTr 15 -g 1 -btz 225 -exp ../exp/MiniImageNet/5way15/1shot/cat -up -uic -fm cat
	python train.py -cTr 15 -g 1 -btz 225 -exp ../exp/MiniImageNet/5way15/1shot/double_weight -up -uic -fm weighted -d

	python train.py -cTr 15 -g 1 -btz 225 -exp ../exp/MiniImageNet/5way15/5shot/unweighted -up -uic  -nsVa 5 -nsTr 5
	python train.py -cTr 15 -g 1 -btz 225 -exp ../exp/MiniImageNet/5way15/5shot/weighted -up -uic -fm weighted -nsVa 5 -nsTr 5
	python train.py -cTr 15 -g 1 -btz 225 -exp ../exp/MiniImageNet/5way15/5shot/cat -up -uic -fm cat -nsVa 5 -nsTr 5
	python train.py -cTr 15 -g 1 -btz 225 -exp ../exp/MiniImageNet/5way15/5shot/double_weight -up -uic -fm weighted -d -nsVa 5 -nsTr 5
elif [ $1 == "Omniglot" ]; 
	then
	echo "training Omniglot experiments";
	python train.py -cTr 15 -btz 225 -data Omniglot -g 1 -exp ../exp/Omniglot/5way15/1shot/unweighted -up -uic 
	python train.py -cTr 15 -btz 225 -data Omniglot -g 1 -exp ../exp/Omniglot/5way15/1shot/weighted -up -uic -fm weighted
	python train.py -cTr 15 -btz 225 -data Omniglot -g 1 -exp ../exp/Omniglot/5way15/1shot/cat -up -uic -fm cat
	python train.py -cTr 15 -btz 225 -data Omniglot -g 1 -exp ../exp/Omniglot/5way15/1shot/double_weight -up -uic -fm weighted -d

	python train.py -cTr 15 -btz 225 -data Omniglot -g 1 -exp ../exp/Omniglot/5way15/5shot/unweighted -up -uic  -nsVa 5 -nsTr 5
	python train.py -cTr 15 -btz 225 -data Omniglot -g 1 -exp ../exp/Omniglot/5way15/5shot/weighted -up -uic -fm weighted -nsVa 5 -nsTr 5
	python train.py -cTr 15 -btz 225 -data Omniglot -g 1 -exp ../exp/Omniglot/5way15/5shot/cat -up -uic -fm cat -nsVa 5 -nsTr 5
	python train.py -cTr 15 -btz 225 -data Omniglot -g 1 -exp ../exp/Omniglot/5way15/5shot/double_weight -up -uic -fm weighted -d -nsVa 5 -nsTr 5
elif [ $1 == "CUB" ]; 
	then
	echo "training CUB experiments";
	python train.py -cTr 15 -btz 225 -data CUB -g 2 -exp ../exp/CUB/5way15/1shot/unweighted -up -uic 
	python train.py -cTr 15 -btz 225 -data CUB -g 2 -exp ../exp/CUB/5way15/1shot/weighted -up -uic -fm weighted
	python train.py -cTr 15 -btz 225 -data CUB -g 2 -exp ../exp/CUB/5way15/1shot/cat -up -uic -fm cat
	python train.py -cTr 15 -btz 225 -data CUB -g 2 -exp ../exp/CUB/5way15/1shot/double_weight -up -uic -fm weighted -d

	python train.py -cTr 15 -btz 225 -data CUB -g 2 -exp ../exp/CUB/5way15/5shot/unweighted -up -uic  -nsVa 5 -nsTr 5
	python train.py -cTr 15 -btz 225 -data CUB -g 2 -exp ../exp/CUB/5way15/5shot/weighted -up -uic -fm weighted -nsVa 5 -nsTr 5
	python train.py -cTr 15 -btz 225 -data CUB -g 2 -exp ../exp/CUB/5way15/5shot/cat -up -uic -fm cat -nsVa 5 -nsTr 5
	python train.py -cTr 15 -btz 225 -data CUB -g 2 -exp ../exp/CUB/5way15/5shot/double_weight -up -uic -fm weighted -d -nsVa 5 -nsTr 5
fi
fi
