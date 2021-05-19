package com.example.travels.entity;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class Province {
    String id;
    String name;
    String tags;
    Integer placecounts;
}
