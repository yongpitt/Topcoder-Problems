#pragma once
#include <iostream>
#include <vector>
#include <hash_set>
#include <hash_map>
#include <queue>
#include <stack>
#include <deque>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <string>
#include <sstream>
#include <fstream>
#include <bitset>

using namespace std;

class TopCoder
{
public:
  TopCoder(void);
	~TopCoder(void);
};

int getmax(int l[], int t[], int n);
int countfriend(int s[], int t[], int n);
int play(std::string costs[], int n);
double bestScore(int rockProb[], int paperProb[], int scissorsProb[], int n);
double bestScore2(int rockProb[], int paperProb[], int scissorsProb[], int n);
int maxWins(int initialLevel, int grezPower[], int n);
int minimalDays(std::string friendship[]);

class GooseInZooDivTwo
{
	struct Node{
	   int x;
	   int y;
	   Node(int a, int b):x(a),y(b){}
	   bool operator== (const Node &b){return x==b.x && y==b.y;}
	   bool operator!= (const Node &b){return x!=b.x || y==b.y;}
	   const Node& operator= (const Node &b){x=b.x; y=b.y; return *this;}
	};
public:
	long long int count(string field[], int n, int dist){
		if(n==0 || field[0].empty())
			return 0;
		int m = field[0].size();
		long long int sum = 0;
		for(int i=0;i<n;i++){
			for(int j=0;j<m;j++){
				if(BFS(i,j,field,n,dist))
					sum++;
			}
		}
		return (1<<sum) - 1;
	}

	bool BFS(int x, int y, string field[], int n, int dist){
		if(field[x][y]=='.' || field[x][y]=='x')
			return false;
		int m = field[0].size();
		queue<Node> q;
		q.push(Node(x,y));
		field[x][y] = 'x';
		while(!q.empty()){
			Node curr = q.front();
			q.pop();
			for(int i=0; i<=dist; i++){
				Node next1(curr.x+i,curr.y+dist-i);
				Node next2(curr.x+i,curr.y-dist+i);
				Node next3(curr.x-i,curr.y+dist-i);
				Node next4(curr.x-i,curr.y-dist+i);
				if(next1.x<n && next1.y<m && field[next1.x][next1.y]!='x' && field[next1.x][next1.y]!='.'){
					field[next1.x][next1.y] = 'x';
				    q.push(next1);
				}
				if(next2.x<n && next2.y>=0 && field[next2.x][next2.y]!='x' && field[next2.x][next2.y]!='.'){
					field[next2.x][next2.y] = 'x';
				    q.push(next2);
				}
				if(next3.x>=0 && next3.y<m && field[next3.x][next3.y]!='x' && field[next3.x][next3.y]!='.'){
					field[next3.x][next3.y] = 'x';
				    q.push(next3);
				}
				if(next4.x>=0 && next4.y>=0 && field[next4.x][next4.y]!='x' && field[next4.x][next4.y]!='.'){
					field[next4.x][next4.y] = 'x';
				    q.push(next4);
				}
		    }	
	    }
		return true;
	}
};




class WolfInZooDivOne{
public:

	int count(int N, string L[], string R[], int n)
	{
		string allL, allR;
		for(int i=0; i<n; i++){
			allL = allL + L[i] + " ";
			allR = allR + R[i] + " ";
		}
		allL.erase(allL.end()-1);
		allR.erase(allR.end()-1);
		vector<int> IL,IR;
		stringstream SL(allL);
		stringstream SR(allR);
		while(!SL.eof()){
			int l, r;
			SL >> l;
			SR >> r;
			IL.push_back(l);
			IR.push_back(r);
		}
		vector<int> dp(N,0);
		
		//collect min info
		vector<int> LM(305,N);
		for(int j=0; j<IR.size(); j++)
			 LM[IR[j]] = min(LM[IR[j]],IL[j]);
	    for(int j=LM.size()-1; j>=0; j--)
			 LM[j] = ((j==LM.size()-1)?LM[j]:min(LM[j],LM[j+1]));
		vector<int> RM(305,0);
		for(int j=0; j<IL.size(); j++)
			 RM[IL[j]] = max(RM[IL[j]],IR[j]);
	    for(int j=0; j<RM.size(); j++)
			 RM[j] = ((j==0)?RM[j]:max(RM[j],RM[j-1]));
				
		dp[0] = 2;
		for(int i=1; i<N; i++){
			int leftMost = LM[i];
			//dp[i] = ((leftMost-1)>=0?dp[leftMost-1]:1)*(i-leftMost+1) + dp[i-1];
			int sum = 0;
			for(int j=leftMost; j<i; j++){
				if(RM[leftMost]<j)
					sum += ((leftMost-1>=0)?dp[leftMost-1]:1);
				else
					sum += ((LM[j]-1>=0)?dp[LM[j]-1]:1);
			}
			sum += ((leftMost-1>=0)?dp[leftMost-1]:1);
			dp[i] = sum + dp[i-1];
		}
		return dp[N-1];
	}
};
