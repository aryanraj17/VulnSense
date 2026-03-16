rule CWE190_IntegerOverflow_MallocMultiply
{
    meta:
        cwe         = "CWE-190"
        severity    = "MEDIUM"
        description = "Multiplication result used directly in malloc without overflow check"

    strings:
        $multiply = /malloc\s*\([^)]*\*[^)]*\)/ ascii
        $safe1    = "SIZE_MAX" ascii
        $safe2    = "INT_MAX" ascii
        $safe3    = "overflow" nocase ascii

    condition:
        $multiply and not any of ($safe*)
}

rule CWE190_IntegerOverflow_Arithmetic
{
    meta:
        cwe         = "CWE-190"
        severity    = "MEDIUM"
        description = "Arithmetic on untrusted input without validation"

    strings:
        $input1 = "argv" ascii
        $input2 = "atoi(" ascii
        $input3 = "strtol(" ascii
        $arith1 = " * " ascii
        $arith2 = " + " ascii
        $check1 = "INT_MAX" ascii
        $check2 = "UINT_MAX" ascii

    condition:
        any of ($input*) and any of ($arith*) and not any of ($check*)
}