package com.example.travels.service;

import com.example.travels.TravelsApplication;
import com.example.travels.entity.User;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringRunner;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest(classes = TravelsApplication.class)
@RunWith(SpringRunner.class)
public class UserServiceImplTest {
    @Autowired
    private UserServiceImpl userService;

    @Test
    public void mytest(){
        User user = new User();
        user.setUsername("消沉");
        user.setPassword("123");
        user.setEmail("123");
        userService.register( user);
    }
}