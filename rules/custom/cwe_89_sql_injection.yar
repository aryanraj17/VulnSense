rule CWE89_SQLInjection_StringConcat
{
    meta:
        cwe         = "CWE-89"
        severity    = "CRITICAL"
        description = "SQL query built with string concatenation"

    strings:
        $sql1    = "SELECT" nocase ascii
        $sql2    = "INSERT" nocase ascii
        $sql3    = "UPDATE" nocase ascii
        $sql4    = "DELETE" nocase ascii
        $concat1 = "strcat(" ascii
        $concat2 = "sprintf(" ascii
        $safe1   = "prepared" nocase ascii
        $safe2   = "parameterized" nocase ascii
        $safe3   = "mysqli_real_escape_string" ascii

    condition:
        any of ($sql*) and
        any of ($concat*) and
        not any of ($safe*)
}

rule CWE89_SQLInjection_UserInput
{
    meta:
        cwe         = "CWE-89"
        severity    = "CRITICAL"
        description = "User input directly used in SQL query"

    strings:
        $input1 = "argv" ascii
        $input2 = "getenv(" ascii
        $input3 = "fgets(" ascii
        $query1 = "SELECT" nocase ascii
        $query2 = "WHERE" nocase ascii

    condition:
        any of ($input*) and any of ($query*)
}