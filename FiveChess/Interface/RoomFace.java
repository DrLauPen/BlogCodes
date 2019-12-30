package Interface;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.*;
import javax.swing.border.Border;

public class RoomFace implements ActionListener{
	JDialog rDialog;
	JLabel rLabel;
	JTextField room;
	JButton okButton;
	JButton cancelButton;
	int roomNum;    //房间号
	
	RoomFace(){
		rDialog = new JDialog();
		rDialog.setTitle("");
		rDialog.setSize(360, 240);
		rDialog.setLayout(null);
		
		JPanel pa = new JPanel();
		pa.setBounds(50, 45, 260, 36);
		pa.setLayout(new GridLayout(1,2));
		pa.setBackground(new Color(228,243,184));
		rLabel = new JLabel();
		rLabel.setText("请输入房间号");
		rLabel.setBackground(new Color(228,243,184));
		rLabel.setFont(new Font("隶书",Font.BOLD,18));
		room = new JTextField();
		pa.add(rLabel);
		pa.add(room);
		rDialog.add(pa);
		
		okButton = new JButton("确定");
		cancelButton = new JButton("返回");
		this.setButton(okButton);
		this.setButton(cancelButton);
		JPanel pb = new JPanel();
		pb.setBounds(100,130,160,36);
		pb.setLayout(new GridLayout(1,2));
		pb.add(okButton);
		pb.add(cancelButton);
		rDialog.add(pb);
		
		JPanel mPanel = new JPanel();
		mPanel.setSize(rDialog.getSize());
		mPanel.setBackground(new Color(228,243,184));
		rDialog.add(mPanel);
		
		okButton.addActionListener(this);
		cancelButton.addActionListener(this);
		
		rDialog.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		rDialog.setLocationRelativeTo(null) ;  
		rDialog.setModal(true);
		rDialog.setVisible(true); 
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		// TODO 自动生成的方法存根
		if(e.getSource() == okButton) {
			if(room.getText().length()==0)
			{
				JOptionPane.showConfirmDialog(rDialog, "用户名为空", "Logout", JOptionPane.OK_CANCEL_OPTION,
						JOptionPane.WARNING_MESSAGE);
				return;
			}else {
				roomNum = Integer.valueOf(room.getText());
				new OnLineFace();
			}
		}else if(e.getSource() == cancelButton) {
			rDialog.dispose();
		}
	}
	
	public void setButton(JButton bu) {
		Border buttonBorder = BorderFactory.createLineBorder(Color.WHITE, 4);
		bu.setFont(new Font("隶书",Font.BOLD,20));
		bu.setBorder(buttonBorder);
		bu.setBackground(new Color(246,236,197));
	}
	
}
