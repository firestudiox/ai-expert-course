import { Link } from "react-router-dom";
import {
  AppBar,
  Container,
  Hidden,
  IconButton,
  List,
  ListItem,
  ListItemText,
  makeStyles,
  Toolbar,
  Fab,
} from "@material-ui/core";
// import Link from "@material-ui/core/Link";
import { Home, KeyboardArrowUp } from "@material-ui/icons";
import * as React from "react";
import HideOnScroll from "./HideOnScroll";
import SideDrawer from "./SideDrawer";
import BackToTop from "./BackToTop";

const useStyles = makeStyles({
  navbarDisplayFlex: {
    display: `flex`,
    justifyContent: `space-between`,
  },
  navListDisplayFlex: {
    display: `flex`,
    justifyContent: `space-between`,
  },
  linkText: {
    textDecoration: `none`,
    textTransform: `uppercase`,
    color: `white`,
  },
});

const navLinks = [
  { title: `Home`, path: `/` },
  { title: `Studio`, path: `/studio` },
  { title: `Words`, path: `/words` },
  { title: `contact`, path: `/contact` },
  { title: `faq`, path: `/faq` },
];

const AppShell = (props) => {
  const classes = useStyles();

  return (
    <>
      <HideOnScroll>
        <AppBar position="fixed">
          <Toolbar component="nav">
            <Container maxWidth="md" className={classes.navbarDisplayFlex}>
              <IconButton edge="start" aria-label="home">
                <a href="/" style={{ color: `white` }}>
                  <Home fontSize="large" />
                </a>
              </IconButton>

              <Hidden smDown>
                <List
                  component="nav"
                  aria-labelledby="main navigation"
                  className={classes.navListDisplayFlex}
                >
                  {navLinks.map(({ title, path }) => (
                    <Link to={path} key={title} className={classes.linkText}>
                      <ListItem button>
                        <ListItemText primary={title} />
                      </ListItem>
                    </Link>
                  ))}
                </List>
              </Hidden>
              <Hidden mdUp>
                <SideDrawer navLinks={navLinks} />
              </Hidden>
            </Container>
          </Toolbar>
        </AppBar>
      </HideOnScroll>
      <Toolbar id="back-to-top-anchor" />

      <BackToTop>
        <Fab color="secondary" size="large" aria-label="scroll back to top">
          <KeyboardArrowUp />
        </Fab>
      </BackToTop>
      <div id="content" style={{ margin: "auto", marginTop: "20px" }}>
        {React.cloneElement(props.children)}
      </div>
    </>
  );
};

export default AppShell;
