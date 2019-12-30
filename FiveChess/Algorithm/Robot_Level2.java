package Algorithm;

import java.util.HashMap;
import java.util.Map;

public class Robot_Level2 extends Robot_Level1 {
	protected float score[][];
	protected float tagRobot = -998;//1000加上该数字即可复原出对应的color类别
	protected float tagPerson = -999;
	protected float RobotPOINT = 20;
	protected float PersonPOINT = 50; 
	Robot_Level2(int color) {
		super(color);
		Set_The_Score();
	}
	public void Set_The_Score() {
		score = new float[super.BORDER][super.BORDER];//创建对应的矩阵
		float half = (float)(super.BORDER-1)/2;
		for(int i=0;i<super.BORDER;i++) {
			for(int j=0;j<super.BORDER;j++) {
				/*对不同的位置计算对应的分值*/
				score[i][j] =-((half-i)*(half-i)+(half-j)*(half-j)); //使用距离公式来作为估值函数,复杂度为O(n^2),越靠近中心则分数越大
			}
		}
	}
	public void Show_Score() {
		/*展示权重矩阵*/
		for(int i=0;i<super.BORDER;i++) {
			for(int j=0;j<super.BORDER;j++) {
				System.out.print((score[i][j])+" ");
			}
			System.out.println();
		}
	}
	public void Get_SelfSteps(int self_i,int self_j) {
		/*传递对应的棋局,并更新*/
		if(self_i==-1) {
			return ;
		}
		score[self_i][self_j] = tagPerson;/*设置该点的分数为固定的tag确保不会重复下*/
		Update_Score(self_i,self_j,PersonPOINT,tagPerson);
	}
	protected void Update_Score(int posi,int posj,float POINT,float tag) {
		/*用于更新被下过的棋局周围的8个位置的对应分数*/
		int i=posi-1;
		int j=posj-1;
		for(int k=0;k<2;k++) {
			/*横向走三步*/
			if(i<super.BORDER && i>=0 && j>=0 && j<super.BORDER) {
				/*如果在范围内*/
				if(score[i][j]!=tagRobot && score[i][j]!=tagPerson)
					score[i][j] += POINT;
			}
			j++;
		}
		/*避免提前跳出需要重新设值*/
		for(int k=0;k<2;k++) {
			/*纵向走2步*/
			if(i<super.BORDER && i>=0 && j>=0 && j<super.BORDER) {
				if(score[i][j]!=tagRobot && score[i][j]!=tagPerson)
					score[i][j] += POINT;
			}
			i++;
		}
		for(int k=0;k<2;k++) {
			/*反向横向走2步*/
			if(i<super.BORDER && i>=0 && j>=0 && j<super.BORDER) {
				if(score[i][j]!=tagRobot && score[i][j]!=tagPerson)
					score[i][j] += POINT;
			}
			j--;
		}
		for(int k=0;k<2;k++) {
			/*反向纵向走一步*/
			if(i<super.BORDER && i>=0 && j>=0 && j<super.BORDER) {
				if(score[i][j]!=tagRobot && score[i][j]!=tagPerson)
					score[i][j] += POINT;
			}
			i--;
		}
	}
	@Override
	public void Generate_Next_Step() {
		/*采用贪心+估值函数实现简单智能*/
		next_vector = new int[2];
		float maxnum = -100000;
		for(int i=0;i<super.BORDER;i++) {
			for(int j=0;j<super.BORDER;j++) {
				/*遍历直接找到最大的值*/
				if(score[i][j]!=tagRobot && score[i][j]!=tagPerson && maxnum<score[i][j]) {
					maxnum = score[i][j];
					next_vector[0] = i;
					next_vector[1] = j;
				}
			}
		}
		Update_Score(next_vector[0],next_vector[1],RobotPOINT,tagRobot);//更改修改的位置,使得存有攻击能力
	}
	@Override
	public void Next_Chess() {
		Generate_Next_Step();
		score[next_vector[0]][next_vector[1]] = tagRobot;//修改当前位置的分数值
		next_i = next_vector[0];
		next_j = next_vector[1];
 	}

}
