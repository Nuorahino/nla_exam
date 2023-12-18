#include <catch2/catch_all.hpp>

SCENARIO( "Verify no argument returns Hello World", "[test_Hello.cpp]" )
{
    REQUIRE( "Hello World" == "Hello World" );
}
