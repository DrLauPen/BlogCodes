package Interface;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JPasswordField;
import javax.swing.JTextField;
import javax.swing.border.Border;

public class RegisterFace implements ActionListener{
	JDialog rframe;
	JLabel label[];
	JPasswordField passwordtext1, passwordtext2;
	JTextField text[];
	JPanel pane, pane2, pane3, pane4, pane5;
	JButton button1, button2;
	JLabel label1;

	RegisterFace() {
		rframe = new JDialog();
		rframe.setTitle("注册");
		rframe.setSize(400, 300);
		
		pane = new JPanel();
		pane2 = new JPanel();
		pane2.setBackground(new Color(228,243,184));
		pane3 = new JPanel();
		pane3.setBackground(new Color(228,243,184));
		pane.setLayout(new GridLayout(6, 2));
		label1 = new JLabel("注册");
		label1.setFont(new Font("隶书",Font.BOLD,20));
		label1.setBackground(new Color(228,243,184));
		button1 = new JButton("确定");
		button2 = new JButton("返回");
		pane2.add(button1);
		pane2.add(button2);
		pane3.add(label1);
		rframe.add(pane3, BorderLayout.NORTH);
		rframe.add(pane2, BorderLayout.SOUTH);
		String str[] = { "用户名 ", "ID  ", "密码  ", "确认密码", "邮箱 " };
		label = new JLabel[str.length];
		for (int i = 0; i < str.length; i++) {
			label[i] = new JLabel(str[i]);
			label[i].setFont(new Font("隶书",Font.BOLD,20));
			label[i].setBackground(new Color(228,243,184));
		}
		text = new JTextField[3];
		for (int i = 0; i < 3; i++) {
			text[i] = new JTextField(10);
			text[i].setSize(10, 5);
		}
		passwordtext1 = new JPasswordField(20);
		passwordtext2 = new JPasswordField(20);
		pane.add(label[0]);
		pane.add(text[0]);
		pane.add(label[1]);
		pane.add(text[1]);
		pane.add(label[2]);
		pane.add(passwordtext1);
		pane.add(label[3]);
		pane.add(passwordtext2);
		pane.add(label[4]);
		pane.add(text[2]);
    	pane.setBackground(new Color(228,243,184));
		rframe.add(pane);
		
		this.setButton(button1);
		this.setButton(button2);
		button1.addActionListener(this);
		button2.addActionListener(this);
		
		rframe.setModal(true);
		rframe.setLocationRelativeTo(null);
		rframe.setVisible(true);
	    rframe.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
	}
	
	
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == button1) {
			int flag = 0;
			String password = new String(passwordtext2.getPassword());
			String password2 = new String(passwordtext1.getPassword());
			if (text[0].getText().length() == 0) {
				flag = 1;
				JOptionPane.showConfirmDialog(rframe, "用户名为空", "Logout", JOptionPane.OK_CANCEL_OPTION,
						JOptionPane.WARNING_MESSAGE);
				return;
			} else if (text[1].getText().length() == 0) {
				flag = 1;
				JOptionPane.showConfirmDialog(rframe, "ID为空", "Logout", JOptionPane.OK_CANCEL_OPTION,
						JOptionPane.WARNING_MESSAGE);
				return;
			}
			if (!password.equals(password2)) {
				flag = 1;
				JOptionPane.showConfirmDialog(rframe, "密码不一致", "Logout", JOptionPane.OK_CANCEL_OPTION,
						JOptionPane.WARNING_MESSAGE);
				return;
			}
			if (text[2].getText().length() == 0) {
				flag = 1;
				JOptionPane.showConfirmDialog(rframe, "邮箱为空", "Logout", JOptionPane.OK_CANCEL_OPTION,
						JOptionPane.WARNING_MESSAGE);
				return;
			}
			
			
		}else { 
			rframe.dispose();
		}
	}
	
	public void setButton(JButton bu) {
		Border buttonBorder = BorderFactory.createLineBorder(Color.WHITE, 4);
		bu.setFont(new Font("隶书",Font.BOLD,20));
		bu.setBorder(buttonBorder);
		bu.setBackground(new Color(246,236,197));
	}
	
}
