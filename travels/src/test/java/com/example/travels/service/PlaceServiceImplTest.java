package com.example.travels.service;

import com.example.travels.TravelsApplication;
import com.example.travels.entity.Place;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringRunner;

import java.util.List;

@SpringBootTest(classes = TravelsApplication.class)
@RunWith(SpringRunner.class)
public class PlaceServiceImplTest {
    @Autowired
    PlaceServiceImpl placeService;

    @Test
    public void testfindByProvinceIdPages(){
        //测试对应的景点查询的方法
        List<Place> byProvinceIdPages = placeService.findByProvinceIdPages(1,3,"1");

        //java流+lambda表达式
        for (Place byProvinceIdPage : byProvinceIdPages) {
            System.out.println(byProvinceIdPage.toString());
        }
    }
}