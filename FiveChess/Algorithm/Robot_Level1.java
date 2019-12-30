package Algorithm;

import java.util.Random;

public class Robot_Level1 extends Player {
	protected int BORDER;
	protected int next_vector[];
	Robot_Level1(int color) {
		super(color);
		this.BORDER = 15;
	}
	public void Generate_Next_Step() {
		next_vector = new int[2];
		Random Random_Generator = new Random();
		next_vector[0] = Random_Generator.nextInt(BORDER);
		next_vector[1] = Random_Generator.nextInt(BORDER);
	}
	@Override
	public void Next_Chess() {
		Generate_Next_Step();
		next_i = next_vector[0];
		next_j = next_vector[1];
	}
}
