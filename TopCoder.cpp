#include "TopCoder.h"

using namespace std;

TopCoder::TopCoder(void)
{
}


TopCoder::~TopCoder(void)
{
}



/*EelAndRabbit*/ 
//http://community.topcoder.com/stat?c=problem_statement&pm=12575&rd=15500
//This problem deals with the maximum interval selection.  Initially I thought I can solve 
//the problem by choosing a time when the number of eels is maximum, adding the number of 
//the overlapped eels (or invervals) at that time, removing the selected intervals and then
//choosing a second time where most intervals overlap among the rest of the intervals.
//To achieve the above algorithm we just need to sort the interval start times and end times
//and maintain a varible numIntervals and update numIntervals based on the entring and leaving 
//intervals as we scan the sorted start and end times from left to right. The sort is O(nlgn)
//and the scan is linear. So it seems that we can derive an O(nlgn) algorithm for this problem.
//However, after thinking a while I realized that getting the maximum number of overlapped 
//intervals each time you select a time may not yeild an overall maximum intervals. See the
//following counter example:
//  -------
//  --        -----
//      -----------
//  --------
//      -----------
//Now we change to another apporach. Still, first sort the intervals based on the starting time.
//An outter loop scans interval i (1 <= i <= n-1), and an inner loop scans interval j (i+1 <= j <= n) 
//When scaning interval j we only count intervals whose start point is later than the start point of 
//interval i to avoid duplicated counts.
int getmax(int l[], int t[], int n)
{
   if(n<=2)
     return n;
   int maxIntervals = 0;
   //sort intervals based on start time (simple bubble sort, can be replaced by a quick sort)
   for(int i=0; i<n-1; i++){
	   for(int j=n-1; j>i; j--){
		   if(t[j]<t[j-1]){
			   swap(l[j],l[j-1]);
		       swap(t[j],t[j-1]);
		   }
	   }
   }
   //l[i] stores the end time
   for(int i=0; i<n; i++)
	   l[i] += t[i];
   //count max intervals
   for(int i=0; i<n-1; i++){
	   //count the intervals covering time t[i]
	   int iintervals = 0;
	   for(int j=0;j<n;j++){
		   if(t[i]<=l[j]&&t[i]>=t[j]){
			   iintervals++;
		   }
		   else if(t[j]>t[i])
			   break;
	   }
	   //count the intervals covering time t[j]
	   for(int j=i+1; j<n; j++){
		   int jintervals = 0;
		   for(int k=i+1;k<n;k++){
			   if(t[j]<=l[k]&&t[j]>=t[k]){
				   jintervals++;
			   }
		   }
		   if(iintervals+jintervals > maxIntervals){
               maxIntervals = iintervals+jintervals;
		   }
	   }
   }
   return maxIntervals;
}


/*ShoutterDiv2*/
//http://community.topcoder.com/stat?c=problem_statement&pm=12578&rd=15500
//This problem can be done with an easy iterative approach.
int countfriend(int s[], int t[], int n)
{
	if(!n) return 0;
	int pairs = 0;

	for(int i=0; i<n-1; i++){
	   for(int j=n-1; j>i; j--){
		   if(s[j]<s[j-1]){
			   swap(s[j],s[j-1]);
		       swap(t[j],t[j-1]);
		   }
	   }
   }

   for(int i=0; i<n-1; i++){
	   for(int j=i+1; j<n; j++){
		   if(s[j]<=t[i])
			   pairs++;
		   else
			   break;
	   }
   }
   return pairs;
}


/*WallGameDiv2*/
//http://community.topcoder.com/stat?c=problem_statement&pm=12579
//DP problem. The trick is that what does it mean for the eel to maximize 
//and rabbit to minimize the moving penalties.  For eel, it can prevent the
//rabbit from moving from an upper row to a lower row in any particular positions
//so that the penalty is maximized. From the rabbit's point of view, the choice
//is that it can move within a row freely as long as it does not encounter a  
//block. So the optimal approach for the rabbit is the move to one specific
//direction (left or right) to the vent (left by eel) that can lead to next row. 
//Based on this idea we can develop the following dynamic programming algorithm:
int play(string costs[], int n)
{
	if(!n) return 0;
	int m = costs[0].size();
	vector<vector<int> > maxCost(n,vector<int>(m,-1));
	//init
	maxCost[0][0] = 0;
	for(int i=1;i<m;i++){
		if(costs[0][i]!='x')
		    maxCost[0][i] = maxCost[0][i-1] + costs[0][i]-'0';
		else
			break;
	}

	for(int i=1;i<n;i++){
		for(int j=0;j<m;j++){
			if(maxCost[i-1][j]!=-1){
				int localCost = maxCost[i-1][j];
				for(int k=j;k>=0;k--){  //move token to the left
					if(costs[i][k]=='x')
						break;
					localCost += (costs[i][k]-'0');
					maxCost[i][k] = max(maxCost[i][k],localCost);
				}
				localCost = maxCost[i-1][j];
				for(int k=j;k<m;k++){  //move token to the right
					if(costs[i][k]=='x')
						break;
					localCost += (costs[i][k]-'0');
					maxCost[i][k] = max(maxCost[i][k],localCost);
				}
			}   
		}
	}
	return maxCost[n-1][m-1];
}



//RockPaperScissors
//http://community.topcoder.com/stat?c=problem_statement&pm=12349
//This problem is a simple accumulative sum of expected scores. A straightforward
//approach is to scan the pobability arrays in a two-level loop nests.
//The outter loop simulates the number of runs and inner loop sum up the expected
//scores based on the remaining dices. The above analysis yeilds the following
//algorithm:
double bestScore(int rockProb[], int paperProb[], int scissorsProb[], int n)
{
	double score = 0;
	if(!n) return score;

	for(int i=0; i<n; i++){
		double rockScore = 0, paperScore = 0, scissorsScore = 0;
		for(int j=i; j<n; j++){
			rockScore += (scissorsProb[j]/300.0)*3.0;
			rockScore += (rockProb[j]/300.0);
			paperScore += (rockProb[j]/300.0)*3.0;
			paperScore += (paperProb[j]/300.0);
			scissorsScore += (paperProb[j]/300.0)*3.0;
			scissorsScore += (scissorsProb[j]/300.0);
		}
		rockScore /= (n-i);
		paperScore /= (n-i);
		scissorsScore /= (n-i);
		score += max(rockScore, max(paperScore,scissorsScore));
	}
	return score;
}
//The above approach has lots of re-computations. Instead of summing up the 
//expected scores at each step, we can, at the very beginning, calculate the
//expected scores over all dices and substract the score due to the current
//dice at each step. Using this idea gives us an O(n) algorithm as follows:
double bestScore2(int rockProb[], int paperProb[], int scissorsProb[], int n)
{
	double score = 0;
	if(!n) return score;

	double rockScore = 0, paperScore = 0, scissorsScore = 0;
	for(int i=0; i<n; i++){		
		rockScore += (scissorsProb[i]/300.0)*3.0;
		rockScore += (rockProb[i]/300.0);
		paperScore += (rockProb[i]/300.0)*3.0;
		paperScore += (paperProb[i]/300.0);
		scissorsScore += (paperProb[i]/300.0)*3.0;
		scissorsScore += (scissorsProb[i]/300.0);
	}
	for(int i=0; i<n; i++){
		score += max(rockScore/(n-i), max(paperScore/(n-i),scissorsScore/(n-i)));
		rockScore -= (scissorsProb[i]/300.0)*3.0;
		rockScore -= (rockProb[i]/300.0);
		paperScore -= (rockProb[i]/300.0)*3.0;
		paperScore -= (paperProb[i]/300.0);
		scissorsScore -= (paperProb[i]/300.0)*3.0;
		scissorsScore -= (scissorsProb[i]/300.0);
	}
	return score;
}



//UndoHistory
//http://community.topcoder.com/stat?c=problem_statement&pm=12523
//The problem statement is somewhat confusing and the most important part in
//addressing this problem is to make sure you understand the specification. It'd
//be clearer to go through all the examples provided there at the official
//website. The post at http://www.bytehood.com/topcoder-srm-579-not-that-tough/430/
//gives a staightforward implementation (comments added for clarity):
string commonPre(string s1, string s2)
{
    string pre = "";
    if(s2.size()<s1.size())
      swap(s1, s2);
    for(int i=0;i<s1.size();i++)
    {
      if(s1[i]==s2[i])
        pre += s1[i];
      else
        break;
    }
    return pre;
}
int minPresses(vector <string> l) {
  
    int r = 0;
    for(int i=0;i<l.size();i++) {
      string pre = "";
      for(int j=0;j<i;j++) {
        string npre = commonPre(l[i], l[j]);
        pre = npre.size()>pre.size()?npre:pre;
      }
      if(pre.size()==0) {  //case when no prefix found
        r += l[i].size()+1;
        if(i>0)
          r+=2;
      }
      else if(i>0 && pre==l[i-1]) {   //case when the whole preceeding string 
        r += l[i].size()-pre.size()+1;//matches a prefix of the current string.
      } 
      else {  //a partial match
        int a = l[i].size()-pre.size()+2+1;
        int b = INT_MAX;
        if(i>0) {
          string npre = commonPre(l[i], l[i-1]);
          if(npre==l[i-1])
            b = l[i].size()-npre.size()+1;
        }
        r += min(a, b);
      }
    }
    return r;
}
//As we can see, the critical and bottleneck part is to find the longest prefix
//of all the piror seen strings that matches the current input string. The above
//algorithm use a brute force approach, resulting in O(n^2) time. Potential 
//improvement can be achieved by maintaining a sorted array of strings that 
//have been processed and the time can be reduced to O(nlgn).


//PrimalUnlicensedCreatures
//http://community.topcoder.com/stat?c=problem_statement&pm=12524
//This is a greedy problem, we always choose the weakest creature (the one 
//with lowest powe level) to beat. This gives us the following algorithm:
int maxWins(int initialLevel, int grezPower[], int n)
{
	if(!n) return 0;
	sort(&grezPower[0], &grezPower[n]);
	int wins = 0, level = initialLevel;
	for(int i=0; i<n; i++){
		if(level<=grezPower[i])
			break;
		else{
			level += grezPower[i]/2;
			wins++;
		}
	}
	return wins;
}


//DancingFoxes
//http://community.topcoder.com/stat?c=problem_statement&pm=12548
//This problem can be viewed as finding a shortest path in a graph. The friendship
//matrix is the adjacency matrix representation of the graph. We can use BFS (breath
//first search) algorithm to traverse the graph and use two arrays to keep track 
//of the visited node and distances from the root (note 0 in this case). When we
//encouter node 1 during the traverse, we know the two foxes are connected though
//some path and the distance - 1 should be the number of dances taken for the two 
//foxes (0 and 1) to become friend.
int minimalDays(string friendship[])
{
	int n = friendship[0].size(); //The input constaint guarantees the array size
	                              //equals to the lenght of one string.
	queue<int> q;
	vector<bool> visited(n,false);
	vector<int> dist(n,std::numeric_limits<int>::max());
    dist[0] = 0;
	q.push(0);
	visited[0] = true;
	while(!q.empty()){
		int curr = q.front();
		q.pop();
		for(int i=0;i<n;i++){
			if(friendship[curr][i]=='Y' && !visited[i]){
				q.push(i);
			    visited[i] = true;
				dist[i]=dist[curr]+1;
				if(i==1) return dist[i]-1;
			}
		}
	}
	return -1;
}



//DeerInZooDivOne
// http://community.topcoder.com/stat?c=problem_statement&pm=12543
/*

A nice discussion on this problem is available at 
http://apps.topcoder.com/wiki/display/tc/SRM+578?showComments=true#comments
At the highest level we need to break the large tree into two subtrees by removing a particular
edge. Loop through each edge to cut the tree into two smaller trees rooted at T1 and T2. 
Since the two largest isomorphic trees may rooted at any node in T1 and T2, we again need a two-level
loop to try each combination of n1 and n2, where n1 belongs to T1 and n2 belongs to T2. For a particular
pair of nodes n1 and n2, we consider their children c11,c12,...,c1n and c21,c22,..c2m. We
can recursively find the size of the largest isomorphic trees rooted at these children notes. Then
we need to pair the children nodes from the n1 and n2 in a way such that the sum of the size of 
the largest isomorphic trees is maximized. Find this size can be solved by the minimum cost flow
algorithm. 

An example solution can be found at
http://community.topcoder.com/stat?c=problem_solution&rm=317251&rd=15498&pm=12543&cr=22694621

#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <map>
#include <list>
#include <queue>
#include <stack>
#include <vector>
#include <string>
#include <utility>
#include <numeric>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <functional>
using namespace std;
 
#define ALL(c) (c).begin(), (c).end()
#define FOR(i, n) for (int i = 0; i < (int)(n); ++i)
#define FOREACH(i, n) for (__typeof(n.begin()) i = n.begin(); i != n.end(); ++i)
#define MEMSET(p, c) memset(p, c, sizeof(p))
typedef long long llint;
typedef pair<int, int> PII;
#ifndef __WATASHI__
#define errf(fmt, ...) do { } while (false)
#endif
 
struct DeerInZooDivOne {
  int getmax(vector <int> a, vector <int> b);
};
 
const int MAXN = 54;
 
vector<int> e[MAXN];
int dp[MAXN][MAXN];
 
int parent[MAXN];
vector<int> child[MAXN];
vector<int> treea, treeb;
 
template<int MAXN, typename S>
struct MinCostMaxFlow {
  struct NegativeCostCircuitExistsException {
  };
 
  struct Edge {
    int v;
    bool c;
    S w;
    int b;
    Edge(int v, bool c, S w, int b) : v(v), c(c), w(w), b(b) { }
  };
 
  int n, source, sink;
  vector<Edge> e[MAXN];
 
  void init(int n, int source, int sink) {
    this->n = n;
    this->source = source;
    this->sink = sink;
    for (int i = 0; i < n; ++i) {
      e[i].clear();
    }
  }
 
  void addEdge(int a, int b, S w) {
    e[a].push_back(Edge(b, true, w, e[b].size()));
    e[b].push_back(Edge(a, false, -w, e[a].size() - 1));  // TODO
  }
 
  bool mark[MAXN];
  S minw[MAXN];
  int dist[MAXN];
  Edge* prev[MAXN];
 
  bool _spfa() {
    queue<int> q;
    fill(mark, mark + n, false);
    fill(minw, minw + n, numeric_limits<S>::max());
    fill(dist, dist + n, 0);
    fill(prev, prev + n, (Edge*)NULL);
    mark[source] = true;
    minw[source] = 0;
 
    q.push(source);
    while (!q.empty()) {
      int cur = q.front();
      mark[cur] = false;
      q.pop();
      for (typename vector<Edge>::iterator it = e[cur].begin(); it != e[cur].end(); ++it) {
        if (!it->c) {
          continue;
        }
        int v = it->v;
        S w = minw[cur] + it->w;
        if (minw[v] > w) { // TODO
          minw[v] = w;
          dist[v] = dist[cur] + 1;
          if (dist[v] >= n) {
            return false;
          }
          prev[v] = &*it;
          if (!mark[v]) {
            mark[v] = true;
            q.push(v);
          }
        }
      }
    }
    return true;
  }
 
  pair<int, S> gao() {
    int sumc = 0;
    S sumw = 0;
    while (true) {
      if (!_spfa()) {
        throw NegativeCostCircuitExistsException();
      } else if (minw[sink] == numeric_limits<S>::max()) {
        break;
      } else {
        ++sumc;
        sumw += minw[sink];
 
        int cur = sink;
        while (cur != source) {
          Edge* e1 = prev[cur];
          e1->c = false;
          Edge* e2 = &e[e1->v][e1->b];
          e2->c = true;
          cur = e2->v;
        }
      }
    }
    return make_pair(sumc, sumw);
  }
};
 
MinCostMaxFlow<MAXN, int> mcmf;
 
int gao(int a, int b) {
  int& ret = dp[a][b];
  if (ret == -1) {
    if (child[a].empty() || child[b].empty()) {
      ret = 1;
    } else {
      int na = child[a].size();
      int nb = child[b].size();
      vector<vector<int> > w(na, vector<int>(nb));
      FOR (i, na) {
        FOR (j, nb) {
          w[i][j] = gao(child[a][i], child[b][j]);
        }
      }
      int s = na + nb;
      int t = s + 1;
      mcmf.init(t + 1, s, t);
      FOR (i, na) {
        mcmf.addEdge(s, i, 0);
      }
      FOR (i, nb) {
        mcmf.addEdge(na + i, t, 0);
      }
      FOR (i, na) {
        FOR (j, nb) {
          mcmf.addEdge(i, na + j, -w[i][j]);
        }
      }
      ret = 1 - mcmf.gao().second;
    }
  }
  return ret;
}
 
void dfs(int v, int p, vector<int>& tree) {
  parent[v] = p;
  child[v].clear();
  tree.push_back(v);
  FOREACH (w, e[v]) {
    if (*w != p) {
      child[v].push_back(*w);
      dfs(*w, v, tree);
    }
  }
}
 
int DeerInZooDivOne::getmax(vector <int> a, vector <int> b) {
  int n = a.size() + 1;
  int ans = 1;
 
  // 50
  FOR (i, a.size()) {
    FOR (j, n) {
      e[j].clear();
    }
    FOR (j, a.size()) {
      if (i != j) {
        e[a[j]].push_back(b[j]);
        e[b[j]].push_back(a[j]);
      }
    }
    treea.clear();
    dfs(a[i], -1, treea);
    treeb.clear();
    dfs(b[i], -1, treeb);
    if (treea.size() < treeb.size()) {
      treea.swap(treeb);
    }
    if ((int)treeb.size() <= ans) {
      continue;
    }
 
    // 25
    FOR (j, treeb.size()) {
      vector<int> dummy;
      dfs(treeb[j], -1, dummy);
      MEMSET(dp, 0xff);
      FOREACH (u, treea) {
        FOREACH (v, treeb) {
          ans = max(ans, gao(*u, *v));
        }
      }
    }
  }
 
  return ans;
}
 
*/



//GooseInZooDivTwo
//http://community.topcoder.com/stat?c=problem_statement&pm=12545
//This problem is essentially a reachability/connectivity problem in graphs. From the problem 
//statement we know for any particular node N in the grid all the other nodes that have the same 
//Manhattan distance forms a node set. Traverse from any node in the node set results in the same
//set, given that the Manhattan distance is fixed. Thus, we can start from a particular node
//in the grid that indicates a goose, do BFS and mark all the visited nodes. Then start from next
//unvisited node do the BFS again. Each time during the BFS, if some new node is visited, we mark
//this BFS as successful. We iteratively do this from the first node in the grid to the last node
//in the grid. The number of successful BFS indiciates how many unique node sets we can derive 
//from the graph (or gird).  Every possible combination of the node sets are also valid. This gives
//us the following algorithm :
/*
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
    //Simple version for illustration purpose, limited to long long int. 
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
};*/



//WolfInZooDivOne
//http://community.topcoder.com/stat?c=problem_statement&pm=12534
/*
This is a nice problem and there are a variety of solutions, either recursive or iterative,
available online. The critical point to address this problem is to realize the fact that
//the road section in which the last wolf is placed is relevant information and should be kept. 
//It's handy to keep this information as it affects the later steps. The following are 
//several example solutions from topcoder:
*/
//The following post provides a sample solution and a nice explanation: 
//http://apps.topcoder.com/wiki/display/tc/SRM+578?showChildren=false
/*
/*
typedef pair<int,int> PII;

vector<PII> V;
int N;
int dp[305][305];

int DP(int last, int at)
{
     if(at > N) return 1;

     int &ret = dp[last][at];
     if(ret != -1) return ret;

     ret = 0;

     //do not place a wolf, it does not violate any rule
     ret = DP(last, at + 1);

     //place a wolf, now progress 'at' to such a place so that, it does not have any scope to break any constraint.
     int sz, end, i;

     sz = V.size();
     end = at;
     for(i = 0; i < sz; i++)
          if(V[i].first <= last && at <= V[i].second)
          {
               if(end < V[i].second)
                    end = V[i].second;
          }

     ret = (ret + DP(at, end + 1))%1000000007;     

     return ret;
}

int count(int _N, vector <string> L, vector <string> R)
{
     int i, a, b;
     string SL, SR;

     N = _N;
     memset(dp, -1, sizeof(dp));
     

     SL = ""; for(i = 0; i < L.size(); i++) SL+=L[i];
     SR = ""; for(i = 0; i < R.size(); i++) SR+=R[i];

     istringstream iSL(SL), iSR(SR);
     V.clear();
     while(iSL>>a)
     {
          iSR>>b;
          a++, b++;
          V.push_back(PII(a,b));
     }

     return DP(0, 1);
}


Another solution that runs in O(m*n):
http://community.topcoder.com/stat?c=problem_solution&rm=317251&rd=15498&pm=12534&cr=22877779
using namespace std;
#define li        long long int
#define rep(i,to) for(li i=0;i<((li)(to));++i)
#define pb        push_back
#define sz(v)     ((li)(v).size())
#define bit(n)    (1ll<<(li)(n))
#define all(vec)  (vec).begin(),(vec).end()
#define each(i,c) for(__typeof((c).begin()) i=(c).begin();i!=(c).end();i++)
#define MP        make_pair
#define F         first
#define S         second
 
const li MAX = 305;
const li mod = 1000000007;
 
class WolfInZooDivOne {
public:
 
  li N;
  
  vector<li> L;
  vector<li> R;
  
  li maxi[MAX];
  li dp[MAX][MAX];
  
  li recur(li pos, li bef)
  {
    //cout << pos << " : " << bef << endl;
    if(pos == N) return 1;
    li &res = dp[pos][bef];
    if(res != -1) return res;
    
    res = recur(pos + 1, bef);
    {
      li next = pos + 1;
      if(bef != MAX - 1) next = max(next, maxi[bef] + 1);
      res = (res + recur(next, pos)) % mod;
    }
    //cout << pos << ", " << bef << endl;
    return res;
  }
  
  void set(vector<string> &s, vector<li> &V)
  {
    V.clear();
    li t;
    stringstream ss;
    rep(i, sz(s)) ss << s[i];
    while(ss >> t) V.pb(t);
  }
 
  int count(int _N, vector <string> _L, vector <string> _R) {
    N = _N;
    set(_L, L);
    set(_R, R);
    memset(maxi, -1, sizeof(maxi));
    rep(i, sz(L)) maxi[L[i]] = R[i];
    rep(i, MAX)if(i) maxi[i] = max(maxi[i], maxi[i - 1]);
    memset(dp, -1, sizeof(dp));
    
    return recur(0, MAX - 1);
  }



An iterative, real dp solution:
//http://community.topcoder.com/stat?c=problem_solution&rm=317251&rd=15498&pm=12534&cr=22694621
using namespace std;
 
#define ALL(c) (c).begin(), (c).end()
#define FOR(i, n) for (int i = 0; i < (int)(n); ++i)
#define FOREACH(i, n) for (__typeof(n.begin()) i = n.begin(); i != n.end(); ++i)
#define MEMSET(p, c) memset(p, c, sizeof(p))
typedef long long llint;
typedef pair<int, int> PII;
#ifndef __WATASHI__
#define errf(fmt, ...) do { } while (false)
#endif
 
struct WolfInZooDivOne {
  int count(int N, vector <string> L, vector <string> R);
};
 
vector<int> parse(const vector<string>& v) {
  vector<int> ret;
  string str;
  FOREACH (i, v) {
    str += *i;
  }
  istringstream iss(str);
  int tmp;
  while (iss >> tmp) {
    ret.push_back(tmp);
  }
  return ret;
}
 
typedef long long llint;
const int MAXN = 303;
const llint MOD = 1000000007;
 
llint dp[MAXN][MAXN];
 
int WolfInZooDivOne::count(int N, vector <string> L, vector <string> R) {
  vector<int> vl = parse(L), vr = parse(R);
  int n = vl.size(), m = N;
 
  vector<int> v;
  FOR (i, m + 1) {
    v.push_back(i - 1);
  }
  FOR (i, n) {
    v[vr[i] + 1] = min(v[vr[i] + 1], vl[i]);
  }
  for (int i = m; i > 0; --i) {
    v[i - 1] = min(v[i - 1], v[i]);
  }
 
  MEMSET(dp, 0);
  for (int i = 1; i <= m; ++i) {
    dp[i][0] = 1;
    for (int j = 1; j < i; ++j) {
      dp[i][j] = 0;
      for (int k = 0; k <= v[i]; ++k) {
        dp[i][j] += dp[j][k];
      }
      dp[i][j] %= MOD;
    }
  }
 
  llint ans = 1;
  FOR (i, MAXN) {
    FOR (j, MAXN) {
      ans += dp[i][j];
    }
  }
  return ans % MOD;
}
*/


//EllysChessboard
//http://community.topcoder.com/stat?c=problem_statement&pm=12527

int minCost(string board[])
{
	
}
