package Algorithm;

import java.util.Scanner;
import java.util.Vector;

public class Player {
	protected int color;
	protected int anothercolor;
	public Vector<Integer> chess;//存储方式是前后两个为一个点坐标
	public int next_i;
	public int next_j;
	Player(int color){
		//构造决定是黑棋还是白棋子
		this.color = color;
		if (color == 1) {
			/*如果是白色*/
			anothercolor = 2;
		}else {
			anothercolor = 1;
		}
		chess = new Vector<Integer>();
		next_i  = -1;
		next_j = -1;
	}
	public int Get_Color(){
		/*获取棋手的颜色,白的为1,黑为2*/
		return color;
	}
	public void Next_Chess(int next_i,int next_j) {
		this.next_i = next_i;
		this.next_j = next_j;
	}	
	public void Next_Chess() {
		/*空*/
	}
	public void Add_Chess() {
		/*当当前的一步可以下了再将next加入到记录中*/
		if (next_i!=-1 && next_j!=-1) {
			chess.add(next_i);
			chess.add(next_j);
		}
	}
}
