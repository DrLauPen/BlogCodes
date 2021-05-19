package com.example.travels.service;

import com.example.travels.TravelsApplication;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringRunner;

@SpringBootTest(classes = TravelsApplication.class)
@RunWith(SpringRunner.class)
@Slf4j
public class ProvinceSeriviceImplTest {
    @Autowired
    ProvinceSeriviceImpl provinceSerivice;

//    @Test
//    public void testFunctions(){
//        List<Province> byPages = provinceSerivice.findByPages(2, 4);
//        //jdk1。8新特性，流遍历
//        byPages.forEach(province -> {
//            log.info(province.toString());
//        });
//        Integer totals = provinceSerivice.findTotals();
//    }

    @Test
    public void testdeleteProvince(){
        //测试删除对应的省份信息
        provinceSerivice.deleteProvince("6");
    }
}