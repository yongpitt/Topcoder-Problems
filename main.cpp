#include <iostream>
#include "TopCoder.h"

using namespace std;

int main()
{
  
	int lml1[] = {2, 4, 3, 2, 2, 1, 10};
	int lmt1[] = {2, 6, 3, 7, 0, 2, 0};
	cout << "Maximum eels the rabbit can catch is: " << getmax(lml1,lmt1,7) << endl;
	
	int lml2[] = {1, 1, 1};
	int lmt2[] = {2, 0, 4};
	cout << "Maximum eels the rabbit can catch is: " << getmax(lml2,lmt2,3) << endl;

	int lml3[] = {1};
	int lmt3[] = {1};
	cout << "Maximum eels the rabbit can catch is: " << getmax(lml3,lmt3,1) << endl;
	
	int lml4[] = {8, 2, 1, 10, 8, 6, 3, 1, 2, 5};
	int lmt4[] = {17, 27, 26, 11, 1, 27, 23, 12, 11, 13};
	cout << "Maximum eels the rabbit can catch is: " << getmax(lml4,lmt4,10) << endl;

	cout << endl;
	
	int s1[] = {1, 2, 4};
	int t1[] = {3, 4, 6};
	cout << "Number of pairs of rabbits become friends: " << countfriend(s1,t1,3) << endl;

	int s2[] = {0};
	int t2[] = {100};
	cout << "Number of pairs of rabbits become friends: " << countfriend(s2,t2,1) << endl;

	int s3[] = {0,0,0};
	int t3[] = {1,1,1};
	cout << "Number of pairs of rabbits become friends: " << countfriend(s3,t3,3) << endl;

	int s4[] = {9,26,8,35,3,58,91,24,10,26,22,18,15,12,15,27,15,60,76,19,12,16,37,35,25,4,22,47,65,3,2,23,26,33,7,11,34,74,67,32,15,45,20,53,60,25,74,13,44,51};
	int t4[] = {26,62,80,80,52,83,100,71,20,73,23,32,80,37,34,55,51,86,97,89,17,81,74,94,79,85,77,97,87,8,70,46,58,70,97,35,80,76,82,80,19,56,65,62,80,49,79,28,75,78};
	cout << "Number of pairs of rabbits become friends: " << countfriend(s4,t4,50) << endl;

	cout << endl;
	string board[] = {"042","391"};
	cout << "Maximum cost: " << play(board,2) << endl;

	string board2[] = {"0xxxx","1x111","1x1x1","11191","xxxx1"};
	cout << "Maximum cost: " << play(board2,5) << endl;

	string board3[] = {"0","5"};
	cout << "Maximum cost: " << play(board3,2) << endl;

    string board4[] = {"0698023477896606x2235481563x59345762591987823x663"
	,"1x88x8338355814562x2096x7x68546x18x54xx1077xx5131"
	,"64343565721335639575x18059738478x156x781476124780"
	,"2139850139989209547489708x3466104x5x3979260330074"
	,"15316x57171x182167994729710304x24339370252x2x8846"
	,"459x479948xx26916349194540891252317x99x4329x34x91"
	,"96x3631804253899x69460666282614302698504342364742"
	,"4x41693527443x7987953128673046855x793298x8747219x"
	,"7735427289436x56129435153x83x37703808694432026643"
	,"340x973216747311x970578x9324423865921864682853036"
	,"x1442314831447x860181542569525471281x762734425650"
	,"x756258910x0529918564126476x481206117984425792x97"
	,"467692076x43x91258xx3xx079x34x29xx916574022682343"
	,"9307x08x451x2882604411x67995x333x045x0x5xx4644590"
	,"4x9x088309856x342242x12x79x2935566358156323631235"
	,"04596921625156134477422x2691011895x8564609x837773"
	,"223x353086929x27222x48467846970564701987061975216"
	,"x4x5887805x89746997xx1419x758406034689902x6152567"
	,"20573059x699979871151x444449x5170122650576586x898"
	,"683344308229681464514453186x51040742xx138x5170x93"
	,"1219892x9407xx63107x24x4914598xx4x478x31485x69139"
	,"856756530006196x8722179365838x9037411399x41126560"
	,"73012x9290145x1764125785844477355xx827269976x4x56"
	,"37x95684445661771730x80xx2x459547x780556228951360"
	,"54532923632041379753304212490929413x377x204659874"
	,"30801x8716360708478705081091961915925276739027360"
	,"5x74x4x39091353819x10x433010250089676063173896656"
	,"03x07174x648272618831383631629x020633861270224x38"
	,"855475149124358107x635160129488205151x45274x18854"
	,"091902044504xx868401923845074542x50x143161647934x"
	,"71215871802698346x390x2570413992678429588x5866973"
	,"87x4538137828472265480468315701832x24590429832627"
	,"9479550007750x658x618193862x80317248236583631x846"
	,"49802902x511965239855908151316389x972x253946284x6"
	,"053078091010241324x8166428x1x93x83809001454563464"
	,"2176345x693826342x093950x12x7290x1186505760xx978x"
	,"x9244898104910492948x2500050208763770568x92514431"
	,"6855xx7x145213846746325656963x0419064369747824511"
	,"88x15690xxx31x20312255171137133511507008114887695"
	,"x391503034x01776xx30264908792724712819642x291x750"
	,"17x1921464904885298x58x58xx174x7x673958x9615x9230"
	,"x9217049564455797269x484428813681307xx85205112873"
	,"19360179004x70496337008802296x7758386170452xx359x"
	,"5057547822326798x0x0569420173277288269x486x582463"
	,"2287970x0x474635353111x85933x33443884726179587xx9"
	,"0x697597684843071327073893661811597376x4767247755"
	,"668920978869307x17435748153x4233659379063530x646x"
	,"0019868300350499779516950730410231x9x18749463x537"
	,"00508xx083203827x42144x147181308668x3192478607467"};

	cout << "Maximum cost: " << play(board4,49) << endl;

	int rockP[] = {100, 100, 100};
	int paperP[] = {100, 100, 100};
	int scissorsP[] = {100, 100, 100};
	double maxScore = bestScore2(rockP,paperP,scissorsP,3);

	int rockP2[] = {300};
	int paperP2[] = {0};
	int scissorsP2[] = {0};
	double maxScore2 = bestScore2(rockP2,paperP2,scissorsP2,1);

	int rockP3[] = {300, 0, 0};
	int paperP3[] = {0, 300, 0};
	int scissorsP3[] = {0, 0, 300};
	double maxScore3 = bestScore2(rockP3,paperP3,scissorsP3,3);

	int rockP4[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                 ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                 ,0,0,0,0,0,0,0,0,0};
	int paperP4[] = {150,300,300,300,300,300,300,300,300,300,300,300,300,300
                 ,300,300,300,300,300,300,300,300,300,300,300,300,300,300
                 ,300,300,300,300,300,300,300,300,300,300,300,300,300,300
                 ,300,300,300,300,300,300,300,300};
	int scissorsP4[] = {150,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                 ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                 ,0,0,0,0,0,0,0,0,0};
	double maxScore4 = bestScore2(rockP4,paperP4,scissorsP4,50);

	int MWA1[] = {10,20,30};
	int MWA2[] = {24, 5, 6, 38};
    int MWA3[] = {3, 3, 3, 3, 3, 1, 25};
	int MWA4[] = {3, 13, 6, 4, 9};
	int MWA5[] = {7, 8, 9, 10};


	int MW1 = maxWins(31,MWA1,3);
	int MW2 = maxWins(20,MWA2,4);
	int MW3 = maxWins(20,MWA3,7);
	int MW4 = maxWins(4,MWA4,5);
	int MW5 = maxWins(7,MWA5,4);


	string MD1[] = {"NNY",
					"NNY",
					"YYN"};
	string MD2[] = {"NNNNN",
					"NNYYY",
					"NYNYY",
					"NYYNY",
					"NYYYN"};
	string MD3[] = {"NNYYNN",
					"NNNNYY",
					"YNNNYN",
					"YNNNNY",
					"NYYNNN",
					"NYNYNN"};
	string MD4[] = {"NNYNNNNYN",
					"NNNNYNYNN",
					"YNNYNYNNN",
					"NNYNYNYYN",
					"NYNYNNNNY",
					"NNYNNNYNN",
					"NYNYNYNNN",
					"YNNYNNNNY",
					"NNNNYNNYN"};
	string MD5[] = {"NY","YN"};

	int mdi1 = minimalDays(MD1);
	int mdi2 = minimalDays(MD2);
	int mdi3 = minimalDays(MD3);
	int mdi4 = minimalDays(MD4);
	int mdi5 = minimalDays(MD5);

	string goose[] = {"vvv"};
	GooseInZooDivTwo gizd;
	int g1 = gizd.count(goose,1,0);

	string goose2[] = {"."};
	int g2 = gizd.count(goose2,1,100);

	string goose3[] = {"vvv"};
	int g3 = gizd.count(goose3,1,1);

	string goose4[] = {"v.v..................v............................"
						,".v......v..................v.....................v"
						,"..v.....v....v.........v...............v......v..."
						,".........vvv...vv.v.........v.v..................v"
						,".....v..........v......v..v...v.......v..........."
						,"...................vv...............v.v..v.v..v..."
						,".v.vv.................v..............v............"
						,"..vv.......v...vv.v............vv.....v.....v....."
						,"....v..........v....v........v.......v.v.v........"
						,".v.......v.............v.v..........vv......v....."
						,"....v.v.......v........v.....v.................v.."
						,"....v..v..v.v..............v.v.v....v..........v.."
						,"..........v...v...................v..............v"
						,"..v........v..........................v....v..v..."
						,"....................v..v.........vv........v......"
						,"..v......v...............................v.v......"
						,"..v.v..............v........v...............vv.vv."
						,"...vv......v...............v.v..............v....."
						,"............................v..v.................v"
						,".v.............v.......v.........................."
						,"......v...v........................v.............."
						,".........v.....v..............vv.................."
						,"................v..v..v.........v....v.......v...."
						,"........v.....v.............v......v.v............"
						,"...........v....................v.v....v.v.v...v.."
						,"...........v......................v...v..........."
						,"..........vv...........v.v.....................v.."
						,".....................v......v............v...v...."
						,".....vv..........................vv.v.....v.v....."
						,".vv.......v...............v.......v..v.....v......"
						,"............v................v..........v....v...."
						,"................vv...v............................"
						,"................v...........v........v...v....v..."
						,"..v...v...v.............v...v........v....v..v...."
						,"......v..v.......v........v..v....vv.............."
						,"...........v..........v........v.v................"
						,"v.v......v................v....................v.."
						,".v........v................................v......"
						,"............................v...v.......v........."
						,"........................vv.v..............v...vv.."
						,".......................vv........v.............v.."
						,"...v.............v.........................v......"
						,"....v......vv...........................v........."
						,"....vv....v................v...vv..............v.."
						,".................................................."
						,"vv........v...v..v.....v..v..................v...."
						,".........v..............v.vv.v.............v......"
						,".......v.....v......v...............v............."
						,"..v..................v................v....v......"
						,".....v.....v.....................v.v......v......."};
	long long int g4 = gizd.count(goose4,50,3);

	string wolfL[] = {"0"};
	string wolfR[] = {"4"};
	WolfInZooDivOne wizd;
	int wi = wizd.count(5,wolfL,wolfR,1);

	string wolfL1[] = {"0 1"};
	string wolfR1[] = {"2 4"};
	int wi1 = wizd.count(5,wolfL1,wolfR1,1);

	string wolfL2[] = {"0 2 2 2 2 2"};
	string wolfR2[] = {"4 3 3 5 5 4"};
	int wi2 = wizd.count(6,wolfL2,wolfR2,1);

	string wolfL3[] = {"0 4 2 7"};
	string wolfR3[] = {"3 9 5 9"};
	int wi3 = wizd.count(10,wolfL3,wolfR3,1);

	string wolfL4[] = {"0 2 2 7 10 1","3 16 22 30 33 38"," 42 44 49 51 57 60 62"," 65 69 72 74 77 7","8 81 84 88 91 93 96"};
	string wolfR4[] = {"41 5 13 22 12 13 ","33 41 80 47 40 ","4","8 96 57 66 ","80 60 71 79"," 70 77 ","99"," 83 85 93 88 89 97 97 98"};
	int wi4 = wizd.count(100,wolfL4,wolfR4,5);

	
	return 0;
}
