package Interface;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import javax.swing.*;
import javax.swing.border.Border;

public class SingleSetFa implements ItemListener{
	JDialog setDialog;
	JRadioButton blackRadio;
	JRadioButton whiteRadio;
	JRadioButton [] diff = new JRadioButton[3];
	JButton exitB;
	int selectChess; //1表示黑色，2表示白色
	int lever;       //难度等级
	
	SingleSetFa(){
		selectChess = 1;   //默认您的棋子为黑色
		lever       = 1;   //默认难度等级为1星
		setDialog = new JDialog();
		setDialog.setTitle("设置");
		setDialog.setBackground(new Color(228,243,184));
		setDialog.setSize(300, 480);
		setDialog.setLayout(null);
		
		JLabel la = new JLabel("您的棋子");
		la.setFont(new Font("隶书",Font.BOLD,20));
		la.setBackground(new Color(228,243,184));
		JPanel pa = new JPanel();
		pa.setBackground(new Color(228,243,184));
		blackRadio = new JRadioButton("黑色");
		whiteRadio = new JRadioButton("白色");
		this.setButton(blackRadio);
		this.setButton(whiteRadio);
		ButtonGroup ag = new ButtonGroup();
		ag.add(blackRadio);
		ag.add(whiteRadio);
		pa.add(blackRadio);
		pa.add(whiteRadio);
		
		JLabel lb = new JLabel("难度选择");
		lb.setFont(new Font("隶书",Font.BOLD,20));
		lb.setBackground(new Color(228,243,184));
		String [] strLb = {"★","★★","★★★"};
		ButtonGroup bg = new ButtonGroup();
		for(int i=0;i<strLb.length;i++) {
			diff[i] = new JRadioButton(strLb[i]);
			this.setButton(diff[i]);
			bg.add(diff[i]);
		}
		
		JPanel pb = new JPanel();
		pb.setBackground(new Color(228,243,184));
		pb.setBounds(50, 60, 220, 320);
		pb.setLayout(new GridLayout(7,1));
		pb.add(la);
		pb.add(pa);
		pb.add(lb);
		for(int i=0;i<diff.length;i++) {
			pb.add(diff[i]);
		}
		
		JButton exitB = new JButton("确定");
		Border buttonBorder = BorderFactory.createLineBorder(Color.WHITE, 4);
		exitB.setFont(new Font("隶书",Font.BOLD,20));
		exitB.setBorder(buttonBorder);
		exitB.setBackground(new Color(246,236,197));
		
		JPanel pc = new JPanel();
		pc.add(exitB);
		pc.setBackground(new Color(228,243,184));
		pb.add(pc);
		setDialog.add(pb);
		blackRadio.setSelected(true);
		diff[0].setSelected(true);
		
		JPanel mPanel = new JPanel();
		mPanel.setSize(setDialog.getSize());
		mPanel.setBackground(new Color(228,243,184));
		setDialog.add(mPanel);
		
		setDialog.setModal(true);
		setDialog.setResizable(false);
		setDialog.setLocationRelativeTo(null);
		setDialog.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		
		blackRadio.addItemListener(this);
		whiteRadio.addItemListener(this);
		for(int i=0;i<diff.length;i++) {
			diff[i].addItemListener(this);
		}
		exitB.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				// TODO 自动生成的方法存根
				setDialog.dispose();
			}
		});
		setDialog.setVisible(true);
	}

	@Override
	public void itemStateChanged(ItemEvent e) {
		// TODO 自动生成的方法存根
		if(blackRadio.isSelected()) {
			this.selectChess = 1;
		}else if(whiteRadio.isSelected()) {
			this.selectChess = 2;
		}
		for(int i=0;i<diff.length;i++) {
			if(diff[i].isSelected()) {
				this.lever = i+1;
				break;
			}
		}
	}
	
	public void setButton(JRadioButton bu) {
		Border buttonBorder = BorderFactory.createLineBorder(Color.WHITE, 4);
		bu.setFont(new Font("隶书",Font.BOLD,20));
		bu.setBorder(buttonBorder);
		bu.setBackground(new Color(228,243,184));
	}
}
